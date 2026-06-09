// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// Kernel-side configuration. This is the MSCCL++ version of
// `DeepEP/csrc/kernels/configs.cuh` with NVSHMEM / IBGDA / mlx5dv includes
// removed so the intranode (NVLink-only) kernels can be built standalone.
// Include this file **only** from `.cu` files.

#pragma once

// CMake defines these. The fallbacks below keep the
// header usable from IDEs and standalone tooling.
#ifndef EP_NVL_PEERS_PER_NODE
#define EP_NVL_PEERS_PER_NODE 8
#endif
#ifndef EP_RDMA_RANKS
#define EP_RDMA_RANKS 2, 4, 8, 16, 32
#endif
#ifndef EP_HIDDEN_SIZES
#define EP_HIDDEN_SIZES 2560, 4096, 5120, 7168
#endif

#define NUM_MAX_FIFO_SLOTS 32768
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#define NUM_WAIT_NANOSECONDS 500

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// Make CLion CUDA indexing work.
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900  // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__     // NOLINT(*-reserved-identifier)
__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) { asm volatile("trap;"); }
#define printf host_device_printf
#endif

// Remove Torch restrictions.
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include "exception.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// NVSHMEM / IBGDA / mlx5dv are only required for the RDMA internode paths and
// are not included here. The internode/low-latency kernels that need them
// will include them directly under `#ifdef MSCCLPP_EP_HAVE_NVSHMEM`.

namespace mscclpp::ep {

template<int... Vs>
struct value_switch {
  static_assert(sizeof...(Vs) > 0, "At least one value must be provided");

  static constexpr int size = sizeof...(Vs);
  static constexpr int max_value = std::max({Vs...});

  // Calls fn with std::integral_constant<int, value> iff value is in Vs.
  template<typename F>
  static void lift(int value, F&& fn) {
    const bool matched = ((value == Vs
      ? (fn(std::integral_constant<int, Vs>{}), true)
      : false) || ...);
    EP_HOST_ASSERT(matched && "Unsupported value");
  }
};

// Equivalent to std::type_identity in C++20
template<typename T>
struct type_identity { using type = T; };

template<auto Tag, typename T>
struct type_case {
  static constexpr auto tag = Tag;
  using type = T;
};

template<typename... Cases>
struct type_switch {
  static_assert(sizeof...(Cases) > 0);

  // Calls fn with type_identity<Cases::type> iff tag is in Cases.
  template<typename TagT, typename F>
  static void lift(TagT tag, F&& fn) {
    // All cases must use the same tag type so equality comparisons are well-defined.
    static_assert((std::is_same_v<decltype(Cases::tag), TagT> && ...),
                  "all cases must share the same tag type");
    const bool matched = ((tag == Cases::tag
      ? (fn(type_identity<typename Cases::type>{}), true)
      : false) || ...);
    EP_HOST_ASSERT(matched && "Unsupported type");
  }
};

namespace detail {
  constexpr int log2_floor(int n) {
    int r = 0;
    while (n > 1) { n /= 2; ++r; }
    return r;
  }

  template<std::size_t... Is>
  constexpr auto make_powers_of_two(std::index_sequence<Is...>) {
    return value_switch<(1 << (Is + 1))...>{};   // 2, 4, 8, ...
  }

  template<int N>
  using powers_of_two_up_to = decltype(make_powers_of_two(std::make_index_sequence<log2_floor(N)>{}));
} // namespace detail

using allowed_nvl_ranks  = detail::powers_of_two_up_to<EP_NVL_PEERS_PER_NODE>;
using allowed_rdma_ranks = value_switch<EP_RDMA_RANKS>;

inline constexpr int num_nvl_peers = allowed_nvl_ranks::max_value;
inline constexpr int max_rdma_ranks = allowed_rdma_ranks::max_value;

static_assert(num_nvl_peers >= 2 && num_nvl_peers <= 8 && (num_nvl_peers & (num_nvl_peers - 1)) == 0,
  "num_nvl_peers must be a power of two in {2, 4, 8}");

// One-byte bitmask: fits any allowed num_nvl_peers ∈ {2, 4, 8}.
using NvlBitmask = std::uint8_t;
static_assert(num_nvl_peers <= 8 * sizeof(NvlBitmask),
              "NvlBitmask must hold num_nvl_peers bits");

// Load type for packed reads of `is_token_in_rank`. Independent of NvlBitmask:
// this matches the byte stride of one RDMA rank's slice in the bool array
// (= num_nvl_peers bytes), to enable a single packed __ldg + warp __shfl_sync.
using PackedNvlBoolsT =
  std::conditional_t<num_nvl_peers == 8, std::uint64_t,
  std::conditional_t<num_nvl_peers == 4, std::uint32_t,
                                         std::uint16_t>>;
static_assert(sizeof(PackedNvlBoolsT) == num_nvl_peers * sizeof(bool),
  "PackedNvlBoolsT must hold one RDMA rank's worth of NVL bools");

using allowed_hidden_sizes = value_switch<EP_HIDDEN_SIZES>;

using supported_dtypes = type_switch<
  type_case<CUDA_R_16BF, nv_bfloat16>,
  type_case<CUDA_R_32F,  float>
>;

// Number of warps in the internode HT dispatch kernel dedicated to copying
// tokens from the user tensor into the RDMA send buffer. The kernel block holds:
//   num_dispatch_rdma_sender_warps   senders
// + 1                                sender coordinator
// + num_nvl_peers                    NVL receivers
// = total warps per block (must fit one cooperative block per SM)
//
// 7 + 1 + num_nvl_peers gives 16 warps (512 threads) at num_nvl_peers=8 - the
// value DeepEP tuned for H100. For num_nvl_peers=4 this is 12 warps (384
// threads), which leaves register room for either:
//   - higher occupancy (2 blocks/SM)                -> keep 7
//   - more sender parallelism in the 16-warp budget -> bump to 11
inline constexpr int num_dispatch_rdma_sender_warps = 7;

// Target total forwarder warps for the internode HT combine kernel. The launcher
// distributes the target across R=kNumRDMARanks via integer floor division with a
// per-rank floor of 1, so the actual forwarder count is:
//
//   kNumForwarderWarps = max(num_combine_forwarder_warps / R, 1) * R
//
// This produces a staircase: kNumForwarderWarps is the largest multiple of R
// not exceeding the target, with a floor of R itself. For target = 20:
//
//   R =  2 ->  20 forwarders  (10 per rank)
//   R =  4 ->  20 forwarders  ( 5 per rank)
//   R =  8 ->  16 forwarders  ( 2 per rank, target underutilized — 8 does not divide 20)
//   R = 16 ->  16 forwarders  ( 1 per rank, floor)
//   R = 32 ->  32 forwarders  ( 1 per rank, floor — target ignored, block grows)
//
// The block layout for an NVL-sender SM holds:
//   num_nvl_peers       NVL senders
// + kNumForwarderWarps  NVL-and-RDMA forwarders
// + 1                   coordinator
// = block warps
//
// The block must fit one cooperative block per SM at this kernel's register
// pressure. DeepEP empirically caps that at ~25 warps (~800 threads) on H100;
// GH200 has the same register file (256 KB / 64K int-registers per SM) so the
// same ceiling applies. Solving num_nvl_peers + kNumForwarderWarps + 1 ≤ 25
// gives an upper bound on the forwarder count of (24 - num_nvl_peers).
//
// The value below is auto-derived from num_nvl_peers to keep block_warps ≤ 25
// (DeepEP's H100/GH200 sweet spot). Rounding to a multiple of 4 keeps the
// staircase plateauing cleanly at small R (the integer-division denominator
// in the per-rank share calculation above):
//
//   num_nvl_peers = 8  ->  16   (matches DeepEP)
//   num_nvl_peers = 4  ->  20
//   num_nvl_peers = 2  ->  20
//
// Override by replacing the formula with a hand-picked literal if you've
// profiled and have a reason; otherwise leave it derived.
inline constexpr int num_combine_forwarder_warps = ((25 - 1 - num_nvl_peers) / 4) * 4;

// Maximum number of distinct RDMA ranks (nodes) a single token can be dispatched
// to. Bounds the size of two register arrays (topk_ranks[], dst_send_buffers[])
// in the internode HT dispatch sender. Tight bound is min(top_k, num_rdma_ranks);
// the routing top-k for DeepSeek-V3 is 8, so DeepEP pins this at 8. Raise it
// if your gate may route a token to more than 8 distinct nodes — the device-side
// assert in dispatch will fire otherwise. Register cost grows linearly.
inline constexpr int max_topk_rdma_ranks = 8;

} // namespace mscclpp::ep
