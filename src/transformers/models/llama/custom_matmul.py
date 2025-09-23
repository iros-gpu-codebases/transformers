from typing import List, Tuple
from torch.utils.cpp_extension import load_inline

import pymongo
import subprocess

CUSTOM_KERNEL_TEMPLATE="""
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

class CustomMatmulManager():
    def __init__(self, threads_per_dim:int=16):
        self.threads_per_dim = threads_per_dim
        self.module_cache = {}
        # setup pymongo connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["idak"]
        self.githash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    def get_module(self, skip_list: List[Tuple[int, int]] = []):
        key = (self.threads_per_dim, tuple(sorted(skip_list)))
        if key not in self.module_cache:
            print(f"Compiling custom matmul kernel with tile size {self.threads_per_dim} and skipping {skip_list}")
            skip_pred = " || ".join([f"(bx == {bx} && by == {by})" for bx, by in skip_list])
            if skip_pred == "":
                skip_pred = "false"
            processed_src = CUSTOM_KERNEL_TEMPLATE.replace("###THREADS_PER_DIM###", str(self.threads_per_dim)).replace("###SKIP_PRED###", skip_pred)
            self.module_cache[key] = {
                "src": processed_src,   
                "kernel": load_inline(
                    name="custom_matmul",
                    cpp_sources="",
                    cuda_sources=processed_src,
                    verbose=True,
                )
            }
            self.db["custom_matmul_kernels"].insert_one({
                "threads_per_dim": self.threads_per_dim,
                "skip_list": skip_list,
                "src": processed_src,
                "githash": self.githash
            })
        return self.module_cache[key]["kernel"]

    def multiply(self, A, B, skip_list: List[Tuple[int, int]] = []):
        assert A.dim() == 2 and B.dim() == 2, "Only 2D tensors supported"
        assert A.shape[1] == B.shape[0], "Incompatible shapes for matmul"
        M = A.shape[0]
        N = A.shape[1]
        # K = B.shape[1] # unused
        blocks_x = (N + self.threads_per_dim - 1) // self.threads_per_dim
        blocks_y = (M + self.threads_per_dim - 1) // self.threads_per_dim
        
        print(blocks_x, blocks_y)
        # check all of skip list in range of blocks
        for bx, by in skip_list:
            assert 0 <= bx < blocks_x and 0 <= by < blocks_y, f"Skip block ({bx}, {by}) out of range ({blocks_x}, {blocks_y})"

        module = self.get_module(skip_list=skip_list)
        return module.matmul_cuda(A, B)

    def save_cache(self):
        all_src = {}
        for key, val in self.module_cache.items():
            all_src[key] = val["src"]
        return all_src