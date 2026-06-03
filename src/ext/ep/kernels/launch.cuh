// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include <utility>

namespace mscclpp::ep {

struct CooperativeLaunchConfig {
  // `attrs` must come before `cfg` so its address is valid by the time `cfg`'s mem-init runs.
  cudaLaunchAttribute attrs[1];
  cudaLaunchConfig_t  cfg;

  CooperativeLaunchConfig(dim3 grid, dim3 block, cudaStream_t stream) noexcept
  : attrs{}, cfg{grid, block, /*dynamicSmemBytes=*/0, stream, attrs, /*numAttrs=*/1} {
    attrs[0].id = cudaLaunchAttributeCooperative;
    attrs[0].val.cooperative = 1;
  }

  // `cfg.attrs` is an interior pointer into this object. Any copy or move
  // would silently produce a dangling/foreign pointer, so forbid both.
  CooperativeLaunchConfig(CooperativeLaunchConfig const&)            = delete;
  CooperativeLaunchConfig(CooperativeLaunchConfig&&)                 = delete;
  CooperativeLaunchConfig& operator=(CooperativeLaunchConfig const&) = delete;
  CooperativeLaunchConfig& operator=(CooperativeLaunchConfig&&)      = delete;

  cudaLaunchConfig_t const* handle() const noexcept { return &cfg; }
};

[[nodiscard]] inline CooperativeLaunchConfig make_cooperative_launch_config(dim3 grid, dim3 block, cudaStream_t stream) noexcept {
  return {grid, block, stream};
}

template <typename... KernelArgs, typename... CallArgs>
void launch_kernel(cudaLaunchConfig_t const& config, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    CUDA_CHECK(cudaLaunchKernelEx(config.handle(), kernel, std::forward<CallArgs>(args)...));
}

template <typename... KernelArgs, typename... CallArgs>
void launch_kernel(dim3 grid, dim3 block, cudaStream_t stream, void (*kernel)(KernelArgs...), CallArgs&&... args) {
  launch_kernel(make_cooperative_launch_config(grid, block, stream), kernel, std::forward<CallArgs>(args)...);
}

} // namespace mscclpp::ep
