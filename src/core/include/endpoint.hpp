// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENDPOINT_HPP_
#define MSCCLPP_ENDPOINT_HPP_

#include <atomic>
#include <memory>
#include <mscclpp/core.hpp>
#include <vector>

#include "ib.hpp"
#include "ofi.hpp"
#include "serialization.hpp"
#include "socket.h"


#define MAX_IF_NAME_SIZE 16

namespace mscclpp {

struct Endpoint::Impl {
  Impl(const EndpointConfig& config, Context::Impl& contextImpl);
  Impl(const std::vector<char>& serialization);

  EndpointConfig config_;
  uint64_t hostHash_;
  uint64_t pidHash_;

  // The following are only used for IB and are undefined for other transports.
  bool ibLocal_;
  bool ibNoAtomic_;
  std::shared_ptr<IbQp> ibQp_;
  IbQpInfo ibQpInfo_;

  // The following are only used for Ethernet and are undefined for other transports.
  std::unique_ptr<Socket> socket_;
  SocketAddress socketAddress_;
  volatile uint32_t* abortFlag_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];

  // The following are only used for Ofi and are undefined for other transports.
  // - ofiResources_ is created for a local endpoint identity
  // - once consumed by OfiConnection, ownership moves to the connection
  // - the endpoint remains only as a serialized identity/wire object
  std::unique_ptr<OfiEndpointResources> ofiResources_;
  OfiEndpointWireInfo ofiWireInfo_;
  std::atomic<bool> ofiConsumed_{false};
};

namespace detail {

inline void serialize(std::vector<char>& buf, const OfiEndpointWireInfo& ofi) {
  serialize(buf, ofi.version);
  serialize(buf, ofi.flags);
  serialize(buf, ofi.addr);
}

inline void serialize(std::vector<char>& buf, const EndpointConfig::Ofi& ofi) {
  serialize(buf, ofi.provider);
  serialize(buf, ofi.domain);
}

inline void serialize(std::vector<char>& buf, const EndpointConfig& cfg) {
  serialize(buf, cfg.transport);
  serialize(buf, cfg.device);
  serialize(buf, cfg.maxWriteQueueSize);
  serialize(buf, cfg.ib);
  serialize(buf, cfg.ofi);
}

inline std::vector<char>::const_iterator deserialize(const std::vector<char>::const_iterator& pos, OfiEndpointWireInfo& ofi) {
  auto cur = deserialize(pos, ofi.version);
  cur = deserialize(cur, ofi.flags);
  cur = deserialize(cur, ofi.addr);
  return cur;
}

inline std::vector<char>::const_iterator deserialize(const std::vector<char>::const_iterator& pos, EndpointConfig::Ofi& ofi) {
  auto cur = deserialize(pos, ofi.provider);
  cur = deserialize(cur, ofi.domain);
  return cur;
}

inline std::vector<char>::const_iterator deserialize(const std::vector<char>::const_iterator& pos, EndpointConfig& cfg) {
  auto cur = deserialize(pos, cfg.transport);
  cur = deserialize(cur, cfg.device);
  cur = deserialize(cur, cfg.maxWriteQueueSize);
  cur = deserialize(cur, cfg.ib);
  cur = deserialize(cur, cfg.ofi);
  return cur;
}

} // namespace detail

}  // namespace mscclpp

#endif  // MSCCLPP_ENDPOINT_HPP_
