// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <boost/lockfree/queue.hpp>
//
#include <libfatbat/controller_base.hpp>
#include <libfatbat/locality.hpp>
#include <libfatbat/logging.hpp>
#include <libfatbat/memory_region.hpp>
#include <libfatbat/memory_segment.hpp>
#include <libfatbat/operation_context_base.hpp>
#include <libfatbat/unique_function.hpp>
//
#include "ofi_controller.hpp"
#include "ofi_opcontext.hpp"

// --------------------------------------------------------------------
MAKE_LOGGER(comm_log, "Comm")

// --------------------------------------------------------------------
// we use this to connect hwmalloc with our code
// --------------------------------------------------------------------
inline libfatbat::memory_segment make_segment(ofi_controller* controller, void* const ptr, std::size_t size,
                                              int device_id) {
  if (controller->get_mrbind()) {
    void* endpoint = controller->get_rx_endpoint().get_ep();
    return libfatbat::memory_segment(controller->get_domain(), ptr, size, true, endpoint, device_id);
  } else {
    return libfatbat::memory_segment(controller->get_domain(), ptr, size, false, nullptr, device_id);
  }
}

// --------------------------------------------------------------------
// A container object for endpoints and message send/recv operations
// --------------------------------------------------------------------
struct ofi_communicator {
  //
  using region_type = libfatbat::memory_region;
  enum class update_signal_result { posted, fallback_to_write, error };
  //
  using context_cache = boost::lockfree::queue<ofi_opcontext*, boost::lockfree::fixed_sized<false>,
                                               boost::lockfree::allocator<std::allocator<void>>>;

  context_cache queue_cache;
  //
  rank_type m_rank = -1;
  rank_type m_size = -1;

 private:
  std::atomic<uint64_t> posted_writes_{0};
  std::atomic<uint64_t> writes_posted_no_cq_{0};
  std::atomic<uint64_t> inbound_signals_seen_{0};
  std::atomic<uint64_t> forwarded_signal_value_{0};
  bool use_write_data_signal_ = false;
  bool prefer_inject_writedata_ = false;
  bool disable_inject_writedata_ = false;
  bool use_tx_counter_flush_ = true;

 public:
  ofi_controller* m_controller;
  libfatbat::endpoint_wrapper m_tx_endpoint;
  libfatbat::endpoint_wrapper m_rx_endpoint;

  // --------------------------------------------------------------------
  ofi_communicator(ofi_controller* controller, rank_type rank, rank_type size)
      : queue_cache(2 * max_callback_queue_size_), m_rank(rank), m_size(size), m_controller(controller) {
    m_tx_endpoint = m_controller->get_tx_endpoint();
    m_rx_endpoint = m_controller->get_rx_endpoint();
    // fill cache with empty request objects taken from the heap so that we can avoid allocations at runtime
    for (std::uint64_t i = 0; i < 2 * max_callback_queue_size_; ++i) {
      queue_cache.push(new ofi_opcontext());
    }
  }

  // --------------------------------------------------------------------
  ~ofi_communicator() {  //
                         // clear_context_caches();
  }

  // --------------------------------------------------------------------
  inline ofi_opcontext* make_operation_context(request_callback_type&& cb) {
    ofi_opcontext* request;
    while (!queue_cache.pop(request)) {
      LIBFATBAT_ERROR(comm_log, "{:<20} {}", "make_ofi_opcontext", "unable to get request from cache");
    }
    request->m_callback = std::move(cb);
    return request;
  }

  // --------------------------------------------------------------------
  rank_type rank() const { return m_rank; }
  rank_type size() const { return m_size; }

  // --------------------------------------------------------------------
  // Match libfatbat test utility naming and semantics for remote address mapping.
  inline uint64_t remote_rma_addr_value(uint64_t remote_base, uint64_t offset = 0) const {
    return m_controller->use_relative_remote_addr() ? offset : (remote_base + offset);
  }

  // --------------------------------------------------------------------
  inline bool use_write_data_signal() const { return use_write_data_signal_; }
  inline bool prefer_inject_writedata() const { return prefer_inject_writedata_; }
  inline bool disable_inject_writedata() const { return disable_inject_writedata_; }
  inline bool use_tx_counter_flush() const { return use_tx_counter_flush_; }

  // --------------------------------------------------------------------
  inline void configure_runtime_options(std::string const& provider, bool localWriteData, bool remoteWriteData,
                                        bool preferInjectWriteData) {
    auto env_enabled = [](char const* name) {
      char const* v = std::getenv(name);
      if (v == nullptr) {
        return false;
      }
      return (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0 ||
              std::strcmp(v, "on") == 0 || std::strcmp(v, "ON") == 0);
    };

    bool const providerIsCxi = provider.find("cxi") != std::string::npos;
    bool const providerIsShm = provider.find("shm") != std::string::npos;
    bool const providerWriteDataEnabled =
        providerIsCxi ? env_enabled("FI_CXI_ENABLE_WRITEDATA")
                      : (providerIsShm || env_enabled("MSCCLPP_OFI_ENABLE_WRITEDATA"));

    use_write_data_signal_ = localWriteData && remoteWriteData && providerWriteDataEnabled;
    use_tx_counter_flush_ = providerIsCxi;
    prefer_inject_writedata_ = preferInjectWriteData;
  }

  // --------------------------------------------------------------------
  inline void disable_write_data_signal() { use_write_data_signal_ = false; }
  inline void mark_inject_writedata_unsupported() { disable_inject_writedata_ = true; }

  // --------------------------------------------------------------------
  inline uint64_t posted_writes() const { return posted_writes_.load(std::memory_order_relaxed); }
  inline uint64_t writes_posted_no_cq() const { return writes_posted_no_cq_.load(std::memory_order_relaxed); }
  inline uint64_t inbound_signals_seen() const { return inbound_signals_seen_.load(std::memory_order_relaxed); }

  // --------------------------------------------------------------------
  inline void record_write_posted_no_cq() {
    posted_writes_.fetch_add(1, std::memory_order_relaxed);
    writes_posted_no_cq_.fetch_add(1, std::memory_order_relaxed);
  }

  // --------------------------------------------------------------------
  inline void rollback_write_posted_no_cq() {
    auto const writes = posted_writes_.load(std::memory_order_relaxed);
    if (writes > 0) {
      posted_writes_.fetch_sub(1, std::memory_order_relaxed);
    }
    auto const nocq = writes_posted_no_cq_.load(std::memory_order_relaxed);
    if (nocq > 0) {
      writes_posted_no_cq_.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  // --------------------------------------------------------------------
  inline uint64_t normalize_remote_cq_data(uint64_t value) {
    if (value == 0 && use_write_data_signal_) {
      value = forwarded_signal_value_.fetch_add(1, std::memory_order_relaxed) + 1;
    } else {
      uint64_t expected = forwarded_signal_value_.load(std::memory_order_relaxed);
      while (value > expected &&
             !forwarded_signal_value_.compare_exchange_weak(expected, value, std::memory_order_relaxed,
                                                            std::memory_order_relaxed)) {
      }
    }
    inbound_signals_seen_.fetch_add(1, std::memory_order_relaxed);
    return value;
  }

  // --------------------------------------------------------------------
  // Match libfatbat test utility behavior, but keep timeout support for connection flush.
  inline void wait_for_write_completions(int64_t timeoutUsec = -1) {
    uint64_t const target = posted_writes();
    if (target == 0) {
      return;
    }

    auto const deadline =
        (timeoutUsec < 0) ? std::chrono::steady_clock::time_point::max()
                          : std::chrono::steady_clock::now() + std::chrono::microseconds(timeoutUsec);

    while (m_controller->get_tx_counter_value() < target) {
      if (timeoutUsec >= 0 && std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("wait_for_write_completions timed out waiting for tx counter");
      }
      std::this_thread::yield();
    }
  }

  // --------------------------------------------------------------------
  // generate a tag with 0xRRRRRRRRtttttttt rank, tag.
  // original tag can be 32bits, then we add 32bits of rank info.
  // Note - this tag setting should not be used without unique context info
  inline std::uint64_t make_tag64(std::uint32_t tag, /*std::uint32_t rank, */ std::uintptr_t ctxt) {
    return (((ctxt & 0x0000'0000'00FF'FFFF) << 24) | ((std::uint64_t(tag) & 0x0000'0000'00FF'FFFF)));
  }

  // --------------------------------------------------------------------
  template <typename Func, typename... Args>
  inline void execute_fi_function(Func F, char const* msg, Args&&... args) {
    ssize_t ret = execute_fi_function_with_status(F, msg, std::forward<Args>(args)...);
    if (ret == 0) {
      return;
    }
    if (ret == -FI_ENOENT) {
      LIBFATBAT_ERROR(comm_log, "{:<20}", "No destination endpoint, terminating.");
      std::terminate();
    }
    if (ret) {
      throw libfatbat::fabric_error(int(ret), msg);
    }
  }

  // --------------------------------------------------------------------
  template <typename Func, typename... Args>
  inline ssize_t execute_fi_function_with_status(Func F, char const* msg, Args&&... args) {
    while (true) {
      ssize_t ret = F(std::forward<Args>(args)...);
      if (ret == -FI_EAGAIN) {
        LIBFATBAT_TRACE(comm_log, "{:<20} Reposting : {}", "FI_EAGAIN", msg);
        m_controller->poll_for_work_completions(this);
        continue;
      }
      return ret;
    }
  }

  // --------------------------------------------------------------------
  // this takes a pinned memory region and sends it using inject instead of send
  void inject_tagged_region(region_type const& send_region, std::size_t size, fi_addr_t rank, uint64_t tag_) {
    LIBFATBAT_DEBUG(comm_log, "{:<20} {} {} tag {} tx endpoint {:p}", "inject tagged", rank, send_region, tag_,
                    (void*)(m_tx_endpoint.get_ep()));
    execute_fi_function(fi_tinject, "fi_tinject", m_tx_endpoint.get_ep(), send_region.get_address(), size, rank,
                        tag_);
  }

  // --------------------------------------------------------------------
  // this takes a pinned memory region and sends it
  void send_tagged_region(region_type const& send_region, std::size_t size, fi_addr_t rank, uint64_t tag_,
                          ofi_opcontext* ctxt) {
    LIBFATBAT_DEBUG(comm_log, "{:<20} {:02} {} tag {} context {:p} tx endpoint {:p}", "send_tagged_region", rank,
                    send_region, tag_, (void*)(ctxt), (void*)(m_tx_endpoint.get_ep()));
    m_controller->sends_posted_++;

    if (size <= m_controller->get_tx_inject_size()) {
      // @todo check reached_recursion_depth() : auto inc = recursion();
      // inject will return immediately, so we do not pass a context, instead return a "ready state"
      inject_tagged_region(send_region, size, rank, tag_);
      // invoke the callback right away since we do not need to poll for completion
      m_controller->sends_complete_++;            
      if (ctxt->m_callback) ctxt->m_callback(rank, tag_);
      return;
    }

    execute_fi_function(fi_tsend, "fi_tsend", m_tx_endpoint.get_ep(), send_region.get_address(), size,
                        send_region.get_local_key(), rank, tag_, ctxt);
  }

  // --------------------------------------------------------------------
  // the receiver posts a single receive buffer to the queue, attaching
  // itself as the context, so that when a message is received
  // the owning receiver is called to handle processing of the buffer
  void recv_tagged_region(region_type const& recv_region, std::size_t size, fi_addr_t rank, uint64_t tag_,
                          ofi_opcontext* ctxt) {
    LIBFATBAT_DEBUG(comm_log, "{:<20} {:02} {} tag {} context {:p} rx endpoint {:p}", "recv_tagged_region", rank,
                    recv_region, tag_, (void*)(ctxt), (void*)(m_rx_endpoint.get_ep()));
    m_controller->recvs_posted_++;
    constexpr uint64_t ignore = 0;
    execute_fi_function(fi_trecv, "fi_trecv", m_rx_endpoint.get_ep(), recv_region.get_address(), size,
                        recv_region.get_local_key(), rank, tag_, ignore, ctxt);
    // if (l.owns_lock()) l.unlock();
  }

  // --------------------------------------------------------------------
  void read_remote(region_type const& recv_region, std::size_t size, fi_addr_t rank, void* remote_addr,
                   uint64_t remote_key, ofi_opcontext* ctxt) {
    LIBFATBAT_DEBUG(comm_log,
                    "{:<20} {:02} {} context {:p} rx endpoint {:p} size {:#10x} rem_addr {:p} rem_key {:#08x}",
                    "read_remote", rank, recv_region, (void*)(ctxt), (void*)(m_tx_endpoint.get_ep()), size,
                    remote_addr, remote_key);
    m_controller->reads_posted_++;
    execute_fi_function(fi_read, "fi_read", m_tx_endpoint.get_ep(), recv_region.get_address(), size,
                        recv_region.get_local_key(), rank, (uint64_t)(remote_addr), remote_key, ctxt);
  }

  // --------------------------------------------------------------------
  void write_remote(region_type const& recv_region, std::size_t size, fi_addr_t rank, void* remote_addr,
                    uint64_t remote_key, ofi_opcontext* ctxt) {
    LIBFATBAT_DEBUG(comm_log,
                    "{:<20} {:02} {} context {:p} rx endpoint {:p} size {:#10x} rem_addr {:p} rem_key {:#08x}",
                    "read_remote", rank, recv_region, (void*)(ctxt), (void*)(m_tx_endpoint.get_ep()), size,
                    remote_addr, remote_key);
    m_controller->writes_posted_++;
    execute_fi_function(fi_write, "fi_write", m_tx_endpoint.get_ep(), recv_region.get_address(), size,
                        recv_region.get_local_key(), rank, (uint64_t)(remote_addr), remote_key, ctxt);
  }

  // --------------------------------------------------------------------
  // Generic write wrapper for cases where the source pointer is offset into a larger MR.
  void write_remote_raw(void* local_buf, std::size_t size, void* local_key, fi_addr_t rank,
                        uint64_t remote_addr, uint64_t remote_key, ofi_opcontext* ctxt,
                        std::function<void()> on_posted = nullptr) {
    LIBFATBAT_DEBUG(comm_log,
                    "{:<20} {:02} local {} context {:p} tx endpoint {:p} size {:#10x} rem_addr {:#018x} "
                    "rem_key {:#08x}",
                    "write_remote_raw", rank, local_buf, (void*)(ctxt), (void*)(m_tx_endpoint.get_ep()), size,
                    remote_addr, remote_key);
    m_controller->writes_posted_++;
    execute_fi_function(fi_write, "fi_write", m_tx_endpoint.get_ep(), local_buf, size, local_key, rank,
                        remote_addr, remote_key, ctxt);
    if (ctxt == nullptr) {
      record_write_posted_no_cq();
    }
    if (on_posted) {
      on_posted();
    }
  }

  // --------------------------------------------------------------------
  int write_data_remote_raw(void* local_buf, std::size_t size, void* local_key, uint64_t imm_data,
                            fi_addr_t rank, uint64_t remote_addr, uint64_t remote_key,
                            ofi_opcontext* ctxt, std::function<void()> on_posted = nullptr) {
    LIBFATBAT_DEBUG(comm_log,
                    "{:<20} {:02} local {} context {:p} tx endpoint {:p} size {:#10x} rem_addr {:#018x} "
                    "rem_key {:#08x} imm_data {:#016x}",
                    "write_data_remote_raw", rank, local_buf, (void*)(ctxt), (void*)(m_tx_endpoint.get_ep()),
                    size, remote_addr, remote_key, imm_data);
    m_controller->writes_posted_++;
    int rc = static_cast<int>(execute_fi_function_with_status(fi_writedata, "fi_writedata",
                                                              m_tx_endpoint.get_ep(), local_buf, size, local_key,
                                                              imm_data, rank, remote_addr, remote_key, ctxt));
    if (rc == 0) {
      if (ctxt == nullptr) {
        record_write_posted_no_cq();
      }
      if (on_posted) {
        on_posted();
      }
      return 0;
    }
    if (rc == -FI_ENOENT) {
      LIBFATBAT_ERROR(comm_log, "{:<20}", "No destination endpoint, terminating.");
      std::terminate();
    }
    return rc;
  }

  // --------------------------------------------------------------------
  int inject_write_raw(void const* local_buf, std::size_t size, fi_addr_t rank, uint64_t remote_addr,
                       uint64_t remote_key, std::function<void()> on_posted = nullptr) {
    LIBFATBAT_DEBUG(comm_log,
                    "{:<20} {:02} local {} tx endpoint {:p} size {:#10x} rem_addr {:#018x} rem_key {:#08x}",
                    "inject_write_raw", rank, local_buf, (void*)(m_tx_endpoint.get_ep()), size, remote_addr,
                    remote_key);
    int rc = static_cast<int>(execute_fi_function_with_status(fi_inject_write, "fi_inject_write",
                                                              m_tx_endpoint.get_ep(), local_buf, size, rank,
                                                              remote_addr, remote_key));
    if (rc == 0) {
      if (on_posted) {
        on_posted();
      }
      return 0;
    }
    if (rc == -FI_ENOENT) {
      LIBFATBAT_ERROR(comm_log, "{:<20}", "No destination endpoint, terminating.");
      std::terminate();
    }
    return rc;
  }

  // --------------------------------------------------------------------
  int inject_writedata_raw(uint64_t imm_data, fi_addr_t rank, uint64_t remote_addr, uint64_t remote_key,
                           std::function<void()> on_posted = nullptr) {
    LIBFATBAT_DEBUG(comm_log,
                    "{:<20} {:02} tx endpoint {:p} rem_addr {:#018x} rem_key {:#08x} imm_data {:#016x}",
                    "inject_writedata_raw", rank, (void*)(m_tx_endpoint.get_ep()), remote_addr, remote_key,
                    imm_data);
    int rc = static_cast<int>(execute_fi_function_with_status(fi_inject_writedata, "fi_inject_writedata",
                                                              m_tx_endpoint.get_ep(), nullptr, 0, imm_data, rank,
                                                              remote_addr, remote_key));
    if (rc == 0) {
      if (on_posted) {
        on_posted();
      }
      return 0;
    }
    if (rc == -FI_ENOENT) {
      LIBFATBAT_ERROR(comm_log, "{:<20}", "No destination endpoint, terminating.");
      std::terminate();
    }
    return rc;
  }

  // --------------------------------------------------------------------
  // Try a writedata-based signal path and decide whether caller should fall back to regular write.
  update_signal_result post_update_signal(void* local_buf, std::size_t size, void* local_key,
                                          fi_addr_t rank, uint64_t remote_addr, uint64_t remote_key,
                                          uint64_t value, bool fail_if_unsupported,
                                          std::function<void()> on_posted = nullptr,
                                          int* error_rc = nullptr) {
    if (!use_write_data_signal_) {
      return update_signal_result::fallback_to_write;
    }

    if (prefer_inject_writedata_ && !disable_inject_writedata_) {
      int rc = inject_writedata_raw(value, rank, remote_addr, remote_key);
      if (rc == 0) {
        return update_signal_result::posted;
      }
      if (rc == -FI_EOPNOTSUPP || rc == -FI_ENOSYS || rc == -FI_EINVAL) {
        mark_inject_writedata_unsupported();
      } else {
        if (error_rc != nullptr) {
          *error_rc = rc;
        }
        return update_signal_result::error;
      }
    }

    int rc = write_data_remote_raw(local_buf, size, local_key, value, rank, remote_addr, remote_key, nullptr,
                     on_posted);
    if (rc == 0) {
      return update_signal_result::posted;
    }

    if (rc == -FI_EOPNOTSUPP || rc == -FI_ENOSYS || rc == -FI_EINVAL) {
      if (fail_if_unsupported) {
        if (error_rc != nullptr) {
          *error_rc = rc;
        }
        return update_signal_result::error;
      }
      disable_write_data_signal();
      return update_signal_result::fallback_to_write;
    }

    if (error_rc != nullptr) {
      *error_rc = rc;
    }
    return update_signal_result::error;
  }

  //   // --------------------------------------------------------------------
  //   ofi_opcontext* read(memory_context::heap_type::pointer const& ptr, std::size_t size, rank_type dst,
  //                           void* remote_addr, uint64_t remote_key, request_callback_type&& cb) {
  //     LIBFATBAT_SCOPE(comm_log, "{} {}", (void*)(this), __func__);

  // #if LIBFATBAT_HAVE_GPU_SUPPORT
  //     auto const& reg = ptr.on_device() ? ptr.device_handle() : ptr.handle();
  // #else
  //     auto const& reg = ptr.handle();
  // #endif

  //     if (cb) {
  //       cb = std::bind(std::move(cb), dst, 0);
  //     }
  //     auto request = make_ofi_opcontext(std::move(cb));

  //     read_remote(reg, size, fi_addr_t(dst), remote_addr, remote_key, request);
  //     return request;
  //   }

  // --------------------------------------------------------------------
  //   ofi_opcontext* send(memory_context::heap_type::pointer const& ptr, std::size_t size, rank_type dst, tag_type
  //   tag,
  //                           request_callback_type&& cb) {
  //     LIBFATBAT_SCOPE(comm_log, "{} {}", (void*)(this), __func__);
  //     std::uint64_t stag = make_tag64(tag, 0);  // this->m_context->get_context_tag());

  // #if LIBFATBAT_HAVE_GPU_SUPPORT
  //     auto const& reg = ptr.on_device() ? ptr.device_handle() : ptr.handle();
  // #else
  //     auto const& reg = ptr.handle();
  // #endif

  //     m_controller->sends_posted_++;

  //     // use optimized inject if msg is very small
  //     if (size <= m_controller->get_tx_inject_size()) {
  //       // @todo check reached_recursion_depth() : auto inc = recursion();
  //       // inject will return immediately, so we do not pass a context, instead return a "ready state"
  //       inject_tagged_region(reg, size, fi_addr_t(dst), stag);
  //       // invoke the callback right away
  //       m_controller->sends_complete_++;
  //       if (cb) cb(dst, tag);
  //       return nullptr;
  //     }

  //     if (cb) {
  //       cb = std::bind(std::move(cb), dst, tag);
  //     }
  //     // construct request which is also an operation context
  //     auto request = make_ofi_opcontext(std::move(cb));

  //     LIBFATBAT_DEBUG(comm_log,
  //                     "{:<20} thisrank {} src/dst {} reg:{} tag {} stag {:#08x} addr {} size {} reg "
  //                     "size {:06} op_ctx {:p} req {:p}",
  //                     "send", rank(), dst, reg, tag, stag, (void*)(reg.get_address()), size, reg.get_size(),
  //                     (void*)request, (void*)request);
  // #if LIBFATBAT_HAVE_GPU_SUPPORT
  //     if (!ptr.on_device()) {
  //       LIBFATBAT_DEBUG(comm_log, "{:<20} mem {}", "send region CRC32",
  //                       libfatbat::log::mem_crc32(reg.get_address(), size));
  //     }
  // #endif

  //     send_tagged_region(reg, size, fi_addr_t(dst), stag, request);
  //     return request;
  //   }

  // --------------------------------------------------------------------
  // progress function that can be called at application level
  progress_status progress() {
    return m_controller->poll_for_work_completions(this);
    // clear_context_caches();
  }

  // --------------------------------------------------------------------
  //   void clear_context_caches()
  //   {
  //     // work through ready callbacks, which were pushed to the queue
  //     // (by other threads)
  //     m_send_cb_queue.consume_all([](ofi_opcontext* req) {
  //       LIBFATBAT_SCOPE(comm_log, "{} {:p}", "m_send_cb_queue.consume_all", (void*) (req));
  //       req->invoke_cb();
  //     });

  //     m_recv_cb_queue.consume_all([](ofi_opcontext* req) {
  //       LIBFATBAT_SCOPE(comm_log, "{} {:p}", "m_recv_cb_queue.consume_all", (void*) (req));
  //       req->invoke_cb();
  //     });
  //   }
};
// --------------------------------------------------------------------
//
// --------------------------------------------------------------------
