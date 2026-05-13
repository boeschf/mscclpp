// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_OFI_HPP_
#define MSCCLPP_OFI_HPP_

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>

//
#include "ofi_communicator.hpp"
#include "ofi_controller.hpp"

using rank_type = std::uint64_t;
using tag_type = std::uint64_t;
using request_callback_type = libfatbat::unique_function<void(rank_type, tag_type)>;

// Forward declarations for libfabric types.
struct fi_info;
struct fid_fabric;
struct fid_domain;
struct fid_ep;
struct fid_av;
struct fid_cq;
struct fid_mr;
struct fid_cntr;

namespace mscclpp {

struct OfiEndpointWireInfo {
  uint32_t version = 1;
  uint64_t rank = 0;
  libfatbat::locality::locality_data addr;
};

struct OfiMemoryAttr {
  libfatbat::mem_Iface iface = libfatbat::mem_Iface::System;
  int device = -1;
  bool deviceOnly = false;
};

struct memregion_deleter {
  void operator()(libfatbat::memory_region *r) {
    if (r)
     r->deregister();
  }
};

using OfiMr = libfatbat::memory_region;
using unique_memregion = std::unique_ptr<OfiMr , memregion_deleter>;

OfiMemoryAttr classifyOfiMemory(void* data);

class OfiCtx {
 public:
  explicit OfiCtx(EndpointConfig::Ofi const& config);
  ~OfiCtx() { }; // closeAll(); }

  OfiCtx(OfiCtx const&) = delete;
  OfiCtx& operator=(OfiCtx const&) = delete;
  OfiCtx(OfiCtx&&) = delete;
  OfiCtx& operator=(OfiCtx&&) = delete;

  ofi_controller* controller() { return controller_.get(); }

  /// Exchange peer locality addresses once and insert them into the OFI address vector.
  void setupPeerConnections();

 private:
  void closeAll() noexcept;

  std::shared_ptr<ofi_controller> controller_;
  EndpointConfig::Ofi config_;
  mutable std::mutex peerAddrMutex_;
  bool peerAddrsInitialized_ = false;
};


class OfiEndpointResources {
 public:
  OfiEndpointResources(OfiCtx& ctx, EndpointConfig const& config);
  ~OfiEndpointResources() { } //  closeAll(); }

  OfiEndpointResources(OfiEndpointResources const&) = delete;
  OfiEndpointResources& operator=(OfiEndpointResources const&) = delete;

  OfiEndpointResources(OfiEndpointResources&&) = delete;
  OfiEndpointResources& operator=(OfiEndpointResources&&) = delete;

  OfiCtx& ctx() const { return *ctx_; }
  inline ofi_controller* controller() const { return ctx_->controller(); }
  inline ofi_communicator* communicator() const { return comm_.get(); }


  fid_ep* ep() const { return ctx_->controller()->get_rx_endpoint().get_ep(); }
  fid_cq* txCq() const { return ctx_->controller()->get_tx_endpoint().get_tx_cq(); }
  fid_cq* rxCq() const { return ctx_->controller()->get_rx_endpoint().get_rx_cq(); }

  libfatbat::locality const& address() const { return ctx_->controller()->here(); }
  uint32_t flags() const { return flags_; }
  bool supportsWriteData() const { return ctx_->controller()->supports_write_data(); }

 private:
  void closeAll() noexcept;
  void cacheAddress();

  OfiCtx* ctx_;
  std::shared_ptr<ofi_communicator> comm_;


  uint32_t flags_ = 0;
};

}  // namespace mscclpp

#endif  // MSCCLPP_OFI_HPP_
