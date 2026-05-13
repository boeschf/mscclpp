// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
//
#include <libfatbat/controller_base.hpp>
#include <libfatbat/logging.hpp>
#include <libfatbat/operation_context_base.hpp>
#include <libfatbat/unique_function.hpp>
//
#include "ofi_opcontext.hpp"

// ------------------------------------------------------------------
MAKE_LOGGER(ofi_ctrl, "OFI_Ctrl")

// --------------------------------------------------------------------
// specialize base controller for MSCCLPP's use case
// --------------------------------------------------------------------
class ofi_controller : public libfatbat::controller_base<ofi_controller, ofi_opcontext> {
 public:
  using remote_cq_data_callback_type = std::function<void(uint64_t)>;

  //
  static std::shared_ptr<ofi_controller> get_instance(std::string const& provider, size_t rank, size_t size,
                                                      size_t threads) {
    static std::shared_ptr<ofi_controller> instance{nullptr};
    if (!instance) {
      instance = std::make_shared<ofi_controller>();
      instance->initialize(provider, rank, size, threads);
    }
    return instance;
  }

  // --------------------------------------------------------------------
  void initialize_derived(std::string const&, size_t, size_t, size_t) {}

  // --------------------------------------------------------------------
  constexpr fi_threading threadlevel_flags() { return FI_THREAD_SAFE; }

  // --------------------------------------------------------------------
  constexpr uint64_t caps_flags(uint64_t /*available_flags*/) const {
    uint64_t flags_required =
        FI_RMA | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RMA_EVENT | FI_LOCAL_COMM;

#ifdef LIBFATBAT_HAVE_GPU_SUPPORT
    flags_required |= FI_HMEM;
#endif

    return flags_required;
  }

  // --------------------------------------------------------------------
  constexpr std::int64_t memory_registration_mode_flags() {
    std::int64_t base_flags = FI_MR_ALLOCATED | FI_MR_LOCAL;

#if defined(LIBFATBAT_HAVE_GPU_SUPPORT)
    base_flags |= FI_MR_HMEM;
#endif

#if defined(LIBFATBAT_HAVE_PROVIDER_CXI)
    return base_flags | FI_MR_ENDPOINT | FI_MR_PROV_KEY;
#endif

    return base_flags;
  }

  // --------------------------------------------------------------------
  constexpr bool needs_cq_counters() { return true; }

  // --------------------------------------------------------------------
  inline int poll_send_queue(fid_cq* tx_cq, void* user_data) {
    return static_cast<controller_base*>(this)->poll_send_queue_default(tx_cq, user_data);
  }

  // --------------------------------------------------------------------
  inline int poll_recv_queue(fid_cq* rx_cq, void* user_data) {
    return static_cast<controller_base*>(this)->poll_recv_queue_default(rx_cq, user_data);
  }

  // --------------------------------------------------------------------
  inline void set_remote_cq_data_callback(remote_cq_data_callback_type cb) {
    remote_cq_data_callback_ = std::move(cb);
  }

  // --------------------------------------------------------------------
  inline void clear_remote_cq_data_callback() { remote_cq_data_callback_ = nullptr; }

  // --------------------------------------------------------------------
  inline void handle_remote_cq_data_completion_impl(uint64_t data) {
    if (remote_cq_data_callback_) {
      remote_cq_data_callback_(data);
    }
  }

 private:
  remote_cq_data_callback_type remote_cq_data_callback_;
};
