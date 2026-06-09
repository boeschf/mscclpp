// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// Intranode runtime helpers. Only the NVLink barrier launcher is ported here
// (see DeepEP `csrc/kernels/runtime.cu::intranode::barrier`). The
// internode/NVSHMEM init helpers are deliberately omitted; the MSCCL++ port
// uses `mscclpp::Bootstrap`/`ProxyService` instead of NVSHMEM.

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace mscclpp {
namespace ep {
namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** task_fifo_ptrs, int head, int rank) {
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

void barrier(int** task_fifo_ptrs, int head, int rank, int num_ranks, cudaStream_t stream) {
  allowed_nvl_ranks::lift(num_ranks, [&](auto N) {
    constexpr int kNumRanks = N;
    launch_kernel(1, 32, stream, barrier<kNumRanks>, task_fifo_ptrs, head, rank);
  });
}

}  // namespace intranode
}  // namespace ep
}  // namespace mscclpp
