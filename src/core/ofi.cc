// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/gpu_utils.hpp>

#include "ofi.hpp"

#include <cstring>
#include <mscclpp/errors.hpp>
#include <cassert>
#include <mutex>

#include "logger.hpp"
//
#include "libfatbat/logging.hpp"
//
#include "ofi_communicator.hpp"
#include "ofi_controller.hpp"

// ------------------------------------------------------------------
MAKE_LOGGER(ofi_dbg, "OFI")
// ------------------------------------------------------------------

namespace mscclpp {


OfiMemoryAttr classifyOfiMemory(void* data) {
  OfiMemoryAttr attr{
    /*.iface = */libfatbat::mem_Iface::System,
    /*.device = */-1,
    /*.deviceOnly = */false
  };
  auto dev = detail::gpuIdFromAddress(data);
  if (dev >=0) {
    attr.device = dev;
#if defined(MSCCLPP_USE_ROCM)
    attr.iface = libfatbat::mem_Iface::Rocr;
#else
    attr.iface = libfatbat::mem_Iface::Cuda;
#endif
    attr.deviceOnly = true;
  }
  return attr;
}


OfiCtx::OfiCtx(EndpointConfig::Ofi const& config) : config_(config) {
  LIBFATBAT_SCOPE(ofi_dbg, "{}", __func__);
#if defined(MSCCLPP_USE_OFI)
  libfatbat::log::init_from_env();

  const int num_threads = 2;  // any N>1 triggers threads safe code paths
  controller_ = ofi_controller::get_instance(config.provider, config.bootstrap->getRank(), config.bootstrap->getNranks(), num_threads);
  setupPeerConnections();
#endif  // defined(MSCCLPP_USE_OFI)
}

void OfiCtx::setupPeerConnections() {
  LIBFATBAT_SCOPE(ofi_dbg, "{}", __func__);
#if defined(MSCCLPP_USE_OFI)
  {
    std::lock_guard<std::mutex> lock(peerAddrMutex_);
    if (peerAddrsInitialized_) {
      INFO(NET, "OfiCtx::setupPeerConnections: already initialized, skipping");
      return;
    }
  }

  auto my_rank = config_.bootstrap->getRank();
  auto nranks = config_.bootstrap->getNranks();
  auto localData = controller_->here().fabric_data();

  INFO(NET, "OfiCtx::setupPeerConnections: begin rank=", my_rank, " nranks=", nranks,
       " provider=", config_.provider, " domain=", config_.domain);

  for (int remoteRank = 0; remoteRank < nranks; ++remoteRank) {
    libfatbat::locality::locality_data remoteData{};
    if (remoteRank == my_rank) {
      remoteData = localData;
    } else {
      const int lo = std::min(my_rank, remoteRank);
      const int hi = std::max(my_rank, remoteRank);
      const int pairTag = lo * nranks + hi;

      INFO(NET, "OfiCtx::setupPeerConnections: exchanging locality with rank ", remoteRank,
           " pair_tag=", pairTag);

      std::vector<char> localWire(sizeof(localData));
      std::memcpy(localWire.data(), localData.data(), localWire.size());
      std::vector<char> remoteWire;

      // Avoid deadlock when bootstrap send/recv are blocking.
      if (my_rank < remoteRank) {
        config_.bootstrap->send(localWire, remoteRank, pairTag);
        config_.bootstrap->recv(remoteWire, remoteRank, pairTag);
      } else {
        config_.bootstrap->recv(remoteWire, remoteRank, pairTag);
        config_.bootstrap->send(localWire, remoteRank, pairTag);
      }

            if (remoteWire.size() != sizeof(remoteData)) {
        THROW(NET, Error, ErrorCode::InvalidUsage,
              "Unexpected OFI locality size from rank ", remoteRank,
              ": got ", remoteWire.size(), " expected ", sizeof(remoteData));
      }
            std::memcpy(remoteData.data(), remoteWire.data(), sizeof(remoteData));
    }

    libfatbat::locality remoteLocality(remoteData, nullptr, static_cast<fi_addr_t>(remoteRank));
    remoteLocality = controller_->insert_address(remoteLocality);
    assert(remoteLocality.fi_address() == static_cast<fi_addr_t>(remoteRank));
    if (remoteRank == my_rank) {
      controller_->setHere(remoteLocality);
      INFO(NET, "OfiCtx::setupPeerConnections: set local address for rank ", my_rank,
           " fi_addr=", static_cast<uint64_t>(remoteLocality.fi_address()));
    }

    INFO(NET, "OfiCtx::setupPeerConnections: inserted rank ", remoteRank,
      " fi_addr=", static_cast<uint64_t>(remoteLocality.fi_address()));

  }

  {
    std::lock_guard<std::mutex> lock(peerAddrMutex_);
    peerAddrsInitialized_ = true;
  }

  INFO(NET, "OfiCtx::setupPeerConnections: complete rank=", my_rank,
       " inserted_ranks=", nranks);
#endif  // defined(MSCCLPP_USE_OFI)
}

OfiEndpointResources::OfiEndpointResources(OfiCtx& ctx, EndpointConfig const& config) : ctx_(&ctx) {
  LIBFATBAT_SCOPE(ofi_dbg, "{}", __func__);
#if defined(MSCCLPP_USE_OFI)
  if (config.transport != Transport::Ofi) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "OfiEndpointResources requires Transport::Ofi");
  }

  try {
    comm_ = std::make_shared<ofi_communicator>(ctx.controller(), config.ofi.bootstrap->getRank(), config.ofi.bootstrap->getNranks());
  } catch (...) {
    throw;
  }
#endif  // defined(MSCCLPP_USE_OFI)
}

}  // namespace mscclpp
