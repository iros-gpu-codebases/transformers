from typing import List, Tuple
from torch.utils.cpp_extension import load_inline

import pymongo
import subprocess
import time 
import torch

NAIVE_FP32_KERNEL_TEMPLATE="""
#include <torch/extension.h>
#include <iostream>

#define THREADS_PER_DIM ###THREADS_PER_DIM###


__device__ __forceinline__ bool should_skip(int bx, int by) {
    return ###SKIP_PRED###;
}

__global__ void MatMulNaive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    if (should_skip(blockIdx.x, blockIdx.y)) {
        return;
    }

    float sum = 0.f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}


torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "A and B must be float32 (torch.float32)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D tensors supported");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix shapes");

    auto A_ = A.contiguous();
    auto B_ = B.contiguous();

    int M = A_.size(0);   // rows of A
    int K = A_.size(1);   // cols of A = rows of B
    int N = B_.size(1);   // cols of B

    auto C = torch::zeros({M, N}, A_.options());

    dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 blocks((N + THREADS_PER_DIM - 1) / THREADS_PER_DIM,
                (M + THREADS_PER_DIM - 1) / THREADS_PER_DIM);

    MatMulNaive<<<blocks, threads>>>(
        A_.data_ptr<float>(),
        B_.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("matmul_cuda", &matmul_cuda, "Shared-memory tiled matmul (CUDA)");
}}
"""

NAIVE_FP16_KERNEL_TEMPLATE="""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <iostream>

#define THREADS_PER_DIM ###THREADS_PER_DIM###

__device__ __forceinline__ bool should_skip(int bx, int by) {
    return ###SKIP_PRED###;
}

__global__ void MatMulNaiveFP16(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    if (should_skip(blockIdx.x, blockIdx.y)) return;

    float sum = 0.f; // accumulate in FP32 for accuracy
    for (int k = 0; k < K; ++k) {
        sum += __half2float(A[row * K + k]) *
               __half2float(B[k * N + col]);
    }
    C[row * N + col] = __float2half(sum);
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "A and B must be float16 (torch.float16)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D tensors supported");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix shapes");

    auto A_ = A.contiguous();
    auto B_ = B.contiguous();

    int M = A_.size(0);
    int K = A_.size(1);
    int N = B_.size(1);

    auto C = torch::zeros({M, N}, A_.options());

    dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);
    dim3 blocks((N + THREADS_PER_DIM - 1) / THREADS_PER_DIM,
                (M + THREADS_PER_DIM - 1) / THREADS_PER_DIM);

    MatMulNaiveFP16<<<blocks, threads>>>(
        reinterpret_cast<const half*>(A_.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "Naive matmul (CUDA, FP16)");
}

"""

WMMA_FP16_MATMUL_TEMPLATE = """
#include <torch/extension.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__device__ __forceinline__ bool should_skip(int bx, int by) {
    return ###SKIP_PRED###;
}

// One 16x16 tile per block; 32 threads (one warp) per block.
__global__ void WMMAMatMulFP16(const half* __restrict__ A,
                               const half* __restrict__ B,
                               half* __restrict__ C,
                               int M, int N, int K) {
    const int tile_m = blockIdx.y;   // tile row
    const int tile_n = blockIdx.x;   // tile col
    const int row0   = tile_m * 16;
    const int col0   = tile_n * 16;

    if (row0 >= M || col0 >= N) return;
    if (should_skip(blockIdx.x, blockIdx.y)) return;

    // Accumulate in FP32.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // A = row-major, B = col-major; both 16x16 steps across K.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    // NOTE: assumes M, N, K are multiples of 16. (Pad otherwise.)
    for (int k0 = 0; k0 < K; k0 += 16) {
        const half* ptrA = A + row0 * K + k0;
        const half* ptrB = B + k0 * N + col0;
        wmma::load_matrix_sync(a_frag, ptrA, K);
        wmma::load_matrix_sync(b_frag, ptrB, N);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // wmma::store_matrix_sync expects a float* for an FP32 accumulator.
    __shared__ float c_tile[16 * 16];
    wmma::store_matrix_sync(c_tile, acc, 16, wmma::mem_row_major);
    __syncthreads();

    // Cast-and-store to half C.
    int t = threadIdx.x; // 0..31
    for (int idx = t; idx < 256; idx += blockDim.x) {
        int r = idx / 16;
        int c = idx % 16;
        int gr = row0 + r;
        int gc = col0 + c;
        if (gr < M && gc < N) {
            C[gr * N + gc] = __float2half(c_tile[idx]);
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "A and B must be float16 (torch.float16)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D tensors supported");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix shapes");

    auto A_ = A.contiguous();
    auto B_ = B.contiguous();

    int M = A_.size(0);
    int K = A_.size(1);
    int N = B_.size(1);

    auto C = torch::zeros({M, N}, A_.options());

    dim3 threads(32, 1, 1);                   // one warp
    dim3 blocks((N + 15) / 16, (M + 15) / 16); // one 16x16 tile per block

    WMMAMatMulFP16<<<blocks, threads>>>(
        reinterpret_cast<const half*>(A_.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "WMMA FP16 matmul (CUDA)");
}
"""


class CustomMatmulManager():
    def __init__(self, threads_per_dim:int=16, use_tensor_cores=True, save_kernels=False):
        self.use_tensor_cores = use_tensor_cores
        self.template_to_use = {
            torch.float32: NAIVE_FP32_KERNEL_TEMPLATE,
            torch.float16: WMMA_FP16_MATMUL_TEMPLATE if use_tensor_cores else NAIVE_FP16_KERNEL_TEMPLATE
        }

        self.threads_per_dim = threads_per_dim
        self.module_cache = {}
        # setup pymongo connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["idak"]
        self.githash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        self.save_kernels = save_kernels
        self.name = f"custom_matmul_{int(time.time()//1)}"

    def get_module(self, dtype=torch.float32, skip_list: List[Tuple[int, int]] = [], verbose=False):
        key = (self.threads_per_dim, tuple(sorted(skip_list)))
        if key not in self.module_cache:
            print(f"Compiling custom matmul kernel with tile size {self.threads_per_dim} and skipping {skip_list[:5]}... ({len(skip_list)} elements)")
            skip_pred = " || ".join([f"(bx == {bx} && by == {by})" for bx, by in skip_list])
            if skip_pred == "":
                skip_pred = "false"
             
            processed_src = self.template_to_use[dtype].replace("###THREADS_PER_DIM###", str(self.threads_per_dim)).replace("###SKIP_PRED###", skip_pred)
            self.module_cache[key] = {
                "src": processed_src,   
                "kernel": load_inline(
                    name=self.name,
                    cpp_sources="",
                    cuda_sources=processed_src,
                    verbose=verbose,
                )
            }
            if self.save_kernels:
                self.db["custom_matmul_kernels"].insert_one({
                    "threads_per_dim": self.threads_per_dim,
                    "skip_list": skip_list,
                    "src": processed_src,
                    "githash": self.githash
                })
        return self.module_cache[key]["kernel"]

    def multiply(self, A, B, skip_list: List[Tuple[int, int]] = [], verbose=False):
        assert A.dtype == B.dtype, f"Data types of A and B dont match: {A.dtype}, {B.dtype}"
        assert A.dtype in self.template_to_use.keys(), f"Matmul impl for dtype={A.dtype} not found."
        assert A.dim() == 2 and B.dim() == 2, "Only 2D tensors supported"
        assert A.shape[1] == B.shape[0], "Incompatible shapes for matmul"
        M = A.shape[0]
        N = A.shape[1]
        # K = B.shape[1] # unused
        blocks_x = (N + self.threads_per_dim - 1) // self.threads_per_dim
        blocks_y = (M + self.threads_per_dim - 1) // self.threads_per_dim

        if verbose:
            print(f"[CustomMatmmulManager] bx={blocks_x}, by={blocks_y}")
        
        # check all of skip list in range of blocks
        for bx, by in skip_list:
            assert 0 <= bx < blocks_x and 0 <= by < blocks_y, f"Skip block ({bx}, {by}) out of range ({blocks_x}, {blocks_y})"

        module = self.get_module(dtype=A.dtype, skip_list=skip_list)
        return module.matmul_cuda(A, B)

    def save_cache(self):
        all_src = {}
        for key, val in self.module_cache.items():
            all_src[key] = val["src"]
        return all_src