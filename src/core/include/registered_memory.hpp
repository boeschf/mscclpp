// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include <optional>
#include <variant>

#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu.hpp>

#include "communicator.hpp"
#include "gpu_ipc_mem.hpp"
#include "ib.hpp"
#include "ofi.hpp"

namespace mscclpp {

namespace detail {

template<TransportTagType T>
struct TransportInfo {};

template<>
struct TransportInfo<TransportTagTrait<Transport::CudaIpc>::tag> {
  GpuIpcMemHandle gpuIpcMemHandle;
};

template<>
struct TransportInfo<IBTransportTag> {
  bool ibLocal;
  const IbMr* ibMr;
  IbMrInfo ibMrInfo;
};

template<>
struct TransportInfo<TransportTagTrait<Transport::Ofi>::tag> {
  bool ofiLocal;
  const OfiMr* ofiMr;
  OfiMrInfo ofiMrInfo;
};

} // namespace detail

template<Transport T>
using TransportInfoType = typename detail::TransportInfo<detail::TransportTagTrait<T>::tag>;

struct TransportInfo {
  Transport transport;

  std::variant<
    std::monostate,
    TransportInfoType<Transport::CudaIpc>,
    detail::TransportInfo<IBTransportTag>,
    TransportInfoType<Transport::Ofi>
  > data;
};

struct RegisteredMemory::Impl {
  // This is the data pointer returned by RegisteredMemory::data(), which may be different from the original data
  // pointer for deserialized remote memory.
  void* data;
  // This is the original data pointer the RegisteredMemory was created with.
  void* originalDataPtr;
  size_t size;
  uint64_t hostHash;
  uint64_t pidHash;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;
  std::shared_ptr<void> peerMemHandle;

  UniqueGpuIpcMemHandle localGpuIpcMemHandle;
  std::shared_ptr<void> remoteMemMap;

  // Only used for IB transport
  std::unordered_map<Transport, std::unique_ptr<const IbMr>> ibMrMap;

  // Only used for OFI transport
  std::unique_ptr<const OfiMr> ofiMr;

  // Optional connection binding. Required for OFI local registrations and used
  // by the communicator to derive the peer for sendMemory/recvMemory.
  //BaseConnection const* boundConnection_ = nullptr;
  std::optional<Connection> boundConnection_;

  Impl(void* data, size_t size, TransportFlags transports, Context::Impl& contextImpl);
  Impl(void* data, size_t size, TransportFlags transports, Context::Impl& contextImpl, Connection const& boundConnection);
  Impl(const std::vector<char>::const_iterator& begin, const std::vector<char>::const_iterator& end);
  /// Constructs a RegisteredMemory::Impl from a vector of data. The constructor should only be used for the remote
  /// memory.
  Impl(const std::vector<char>& data);

  void bindConnection(Connection const& connection) {
    if (boundConnection_.has_value()) {
      throw Error("RegisteredMemory is already bound to a connection", ErrorCode::InvalidUsage);
    }
    boundConnection_ = connection;
  }

  const TransportInfo& getTransportInfo(Transport transport) const;

  void registerNonOfiLocalTransports(void* data, size_t size, TransportFlags transports, Context::Impl& contextImpl);
};

}  // namespace mscclpp

#endif  // MSCCLPP_REGISTERED_MEMORY_HPP_
