#pragma once

#include <cstdint>
#include <memory>
//
#include <libfatbat/logging.hpp>
#include <libfatbat/operation_context_base.hpp>
#include <libfatbat/unique_function.hpp>

// ------------------------------------------------------------------
MAKE_LOGGER(ofi_ctxt, "OFI_OpCtxt")
//
using rank_type = std::uint64_t;
using tag_type = std::uint64_t;
using request_callback_type = libfatbat::unique_function<void(rank_type, tag_type)>;
//
const std::uint64_t max_callback_queue_size_ = 1024;

// --------------------------------------------------------------------
// we are not supporting cancellation for now
// --------------------------------------------------------------------
struct ofi_opcontext : public libfatbat::operation_context_base<ofi_opcontext> {

  // when the operation completes, this callback is invoked to trigger user defined actions
  request_callback_type m_callback;

  // --------------------------------------------------------------------
  ofi_opcontext() : libfatbat::operation_context_base<ofi_opcontext>(), m_callback(nullptr) {
    // LIBFATBAT_SCOPE("{} {}", (void*) (this), __func__);
  }

  // --------------------------------------------------------------------
  inline void invoke_cb() {
    if (m_callback) m_callback(0, 0);
  }

  // --------------------------------------------------------------------
  // When a completion returns FI_ECANCELED, this is called
  inline int handle_cancelled() {
    LIBFATBAT_SCOPE(ofi_ctxt, "{} {}", (void*)(this), __func__);
    invoke_cb();
    return 1;
  }

  // --------------------------------------------------------------------
  // Called when a tagged recv completes
  inline int handle_tagged_recv_completion_impl(void* user_data) {
    LIBFATBAT_SCOPE(ofi_ctxt, "{} {} user_data {}", (void*)(this), __func__, user_data);
    invoke_cb();
    return 1;
  }

  // --------------------------------------------------------------------
  // Called when a tagged send completes
  inline int handle_tagged_send_completion_impl(void* user_data) {
    LIBFATBAT_SCOPE(ofi_ctxt, "{} {} user_data {}", (void*)(this), __func__, user_data);
    invoke_cb();
    return 1;
  }

  // --------------------------------------------------------------------
  // Called when an RMA read completes
  inline int handle_rma_read_completion_impl() {
    LIBFATBAT_SCOPE(ofi_ctxt, "{} {}", (void*)(this), __func__);
    invoke_cb();
    return 1;
  }

  // --------------------------------------------------------------------
  // Called when an RMA write completes
  inline int handle_rma_write_completion_impl() {
    LIBFATBAT_SCOPE(ofi_ctxt, "{} {}", (void*)(this), __func__);
    invoke_cb();
    return 1;
  }
};
