// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context.hpp"

#include <mscclpp/env.hpp>
#include <sstream>

#include "api.h"
#include "connection.hpp"
#include "debug.h"
#include "endpoint.hpp"
#include "registered_memory.hpp"

namespace mscclpp {

CudaIpcStream::CudaIpcStream(int deviceId)
    : stream_(std::make_shared<CudaStreamWithFlags>()), deviceId_(deviceId), dirty_(false) {}

void CudaIpcStream::setStreamIfNeeded() {
  if (!env()->cudaIpcUseDefaultStream && stream_->empty()) {
    stream_->set(cudaStreamNonBlocking);
  }
}

void CudaIpcStream::memcpyD2D(void* dst, const void* src, size_t nbytes) {
  CudaDeviceGuard deviceGuard(deviceId_);
  setStreamIfNeeded();
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, *stream_));
  dirty_ = true;
}

void CudaIpcStream::memcpyH2D(void* dst, const void* src, size_t nbytes) {
  CudaDeviceGuard deviceGuard(deviceId_);
  setStreamIfNeeded();
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, *stream_));
  dirty_ = true;
}

void CudaIpcStream::sync() {
  if (dirty_) {
    CudaDeviceGuard deviceGuard(deviceId_);
    setStreamIfNeeded();
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(*stream_));
    dirty_ = false;
  }
}

IbCtx* Context::Impl::getIbContext(Transport ibTransport) {
  // Find IB context or create it
  auto it = ibContexts_.find(ibTransport);
  if (it == ibContexts_.end()) {
    auto ibDev = getIBDeviceName(ibTransport);
    ibContexts_[ibTransport] = std::make_unique<IbCtx>(ibDev);
    return ibContexts_[ibTransport].get();
  }
  return it->second.get();
}

void Context::Impl::bindOfiConfig(EndpointConfig::Ofi const& ofiConfig) {
  std::lock_guard<std::mutex> lock(ofiMutex_);
  if (!hasOfiConfig_) {
    ofiConfig_ = ofiConfig;
    hasOfiConfig_ = true;
    return;
  }

  // If no OFI context has been created yet, allow rebinding after earlier setup failures.
  if (!ofiCtx_) {
    ofiConfig_ = ofiConfig;
    return;
  }

  if (ofiConfig_.provider != ofiConfig.provider || ofiConfig_.domain != ofiConfig.domain) {
    throw Error("All OFI endpoints in one Context must use the same provider/domain",
                ErrorCode::InvalidUsage);
  }
}

std::unique_ptr<OfiEndpointResources> Context::Impl::createOfiEndpointResources(EndpointConfig const& config) {
  std::lock_guard<std::mutex> lock(ofiMutex_);
  if (!hasOfiConfig_) {
    throw Error("No OFI configuration is bound to this Context", ErrorCode::InvalidUsage);
  }
  INFO(MSCCLPP_NET, "Context::Impl::createOfiEndpointResources provider=%s domain=%s",
       ofiConfig_.provider.c_str(), ofiConfig_.domain.c_str());
  if (!ofiCtx_) {
    INFO(MSCCLPP_NET, "Creating OfiCtx for Context provider=%s domain=%s",
         ofiConfig_.provider.c_str(), ofiConfig_.domain.c_str());
    ofiCtx_ = std::make_unique<OfiCtx>(ofiConfig_);
  }
  INFO(MSCCLPP_NET, "Returning OfiEndpointResources for Context provider=%s domain=%s",
       ofiConfig_.provider.c_str(), ofiConfig_.domain.c_str());
  return std::make_unique<OfiEndpointResources>(*ofiCtx_, config);
}

std::shared_ptr<uint64_t> Context::Impl::getToken() {
  if (!tokenPool_) {
    tokenPool_ = std::make_shared<TokenPool>(maxNumTokens_);
  }
  return tokenPool_->getToken();
}

MSCCLPP_API_CPP Context::Context() : pimpl_(std::make_unique<Impl>()) {}

MSCCLPP_API_CPP Context::~Context() = default;

MSCCLPP_API_CPP RegisteredMemory Context::registerMemory(void* ptr, size_t size, TransportFlags transports) {
  return RegisteredMemory(std::make_shared<RegisteredMemory::Impl>(ptr, size, transports, *pimpl_));
}

MSCCLPP_API_CPP RegisteredMemory Context::registerMemory(void* ptr, size_t size, TransportFlags transports,
                                                         const Connection& connection) {
  return RegisteredMemory(std::make_shared<RegisteredMemory::Impl>(ptr, size, transports, *pimpl_, connection));
}

MSCCLPP_API_CPP Endpoint Context::createEndpoint(EndpointConfig config) {
  INFO(MSCCLPP_NET, "Context::createEndpoint transport=%d", static_cast<int>(config.transport));
  return Endpoint(std::make_shared<Endpoint::Impl>(config, *pimpl_));
}

MSCCLPP_API_CPP Connection Context::connect(const Endpoint& localEndpoint, const Endpoint& remoteEndpoint) {
  if (localEndpoint.device().type == DeviceType::GPU && localEndpoint.device().id < 0) {
    throw Error("No GPU device ID provided for local endpoint", ErrorCode::InvalidUsage);
  }
  if (remoteEndpoint.device().type == DeviceType::GPU && remoteEndpoint.device().id < 0) {
    throw Error("No GPU device ID provided for remote endpoint", ErrorCode::InvalidUsage);
  }
  auto localTransport = localEndpoint.transport();
  auto remoteTransport = remoteEndpoint.transport();
  if (localTransport != remoteTransport &&
      !(AllIBTransports.has(localTransport) && AllIBTransports.has(remoteTransport))) {
    std::stringstream ss;
    ss << "Transport mismatch between local (" << localTransport << ") and remote (" << remoteEndpoint.transport()
       << ") endpoints";
    throw Error(ss.str(), ErrorCode::InvalidUsage);
  }
  std::shared_ptr<BaseConnection> conn;
  if (localTransport == Transport::CudaIpc) {
    conn = std::make_shared<CudaIpcConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else if (AllIBTransports.has(localTransport)) {
    conn = std::make_shared<IBConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else if (localTransport == Transport::Ethernet) {
    conn = std::make_shared<EthernetConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else if (localTransport == Transport::Ofi) {
    conn = std::make_shared<OfiConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else {
    throw Error("Unsupported transport", ErrorCode::InternalError);
  }
  return Connection(conn);
}

}  // namespace mscclpp
