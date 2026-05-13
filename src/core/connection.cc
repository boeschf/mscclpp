// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "connection.hpp"
#include <memory>
#include <stdexcept>

#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/utils.hpp>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <thread>

#include "api.h"
#include "atomic.hpp"
#include "context.hpp"
#include "endpoint.hpp"
#include "gpu_utils_internal.hpp"
#include "logger.hpp"
#include "ofi.hpp"


namespace mscclpp {

static void validateTransport(RegisteredMemory mem, Transport transport, uint64_t offset = 0, uint64_t size = 0) {
  if (!mem.transports().has(transport)) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "RegisteredMemory does not support this transport");
  }
  if (offset + size > mem.size()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "RegisteredMemory out of bounds");
  }
}

static bool isSameProcess(const Endpoint& a, const Endpoint& b) {
  return a.hostHash() == b.hostHash() && a.pidHash() == b.pidHash();
}

// BaseConnection

Endpoint::Impl& BaseConnection::getImpl(Endpoint& endpoint) { return *endpoint.pimpl_; }

const Endpoint::Impl& BaseConnection::getImpl(const Endpoint& endpoint) { return *(endpoint.pimpl_); }

const RegisteredMemory::Impl& BaseConnection::getImpl(const RegisteredMemory& memory) { return *(memory.pimpl_); }

Context::Impl& BaseConnection::getImpl(Context& context) { return *(context.pimpl_); }

MSCCLPP_API_CPP BaseConnection::BaseConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint)
    : context_(context), localEndpoint_(localEndpoint), maxWriteQueueSize_(localEndpoint.maxWriteQueueSize()) {}

MSCCLPP_API_CPP std::shared_ptr<Context> BaseConnection::context() const { return context_; }

MSCCLPP_API_CPP const Device& BaseConnection::localDevice() const { return localEndpoint_.device(); }

MSCCLPP_API_CPP int BaseConnection::getMaxWriteQueueSize() const { return maxWriteQueueSize_; }

unique_memregion BaseConnection::registerOfiMr(void* data, size_t size) const {
  (void)data;
  (void)size;
  THROW(CONN, Error, ErrorCode::InvalidUsage, "This connection does not support endpoint-scoped OFI MR registration");
}

// Connection wrapper

Connection::Connection(std::shared_ptr<BaseConnection> impl) : impl_(impl) {}

MSCCLPP_API_CPP void Connection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
                                       uint64_t srcOffset, uint64_t size) {
  impl_->write(dst, dstOffset, src, srcOffset, size);
}

MSCCLPP_API_CPP void Connection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src,
                                               uint64_t newValue) {
  impl_->updateAndSync(dst, dstOffset, src, newValue);
}

MSCCLPP_API_CPP void Connection::flush(int64_t timeoutUsec) { impl_->flush(timeoutUsec); }

MSCCLPP_API_CPP Transport Connection::transport() const { return impl_->transport(); }

MSCCLPP_API_CPP Transport Connection::remoteTransport() const { return impl_->remoteTransport(); }

MSCCLPP_API_CPP std::shared_ptr<Context> Connection::context() const { return impl_->context(); }

MSCCLPP_API_CPP const Device& Connection::localDevice() const { return impl_->localDevice(); }

MSCCLPP_API_CPP int Connection::getMaxWriteQueueSize() const { return impl_->getMaxWriteQueueSize(); }

// CudaIpcConnection

CudaIpcConnection::CudaIpcConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                                     const Endpoint& remoteEndpoint)
    : BaseConnection(context, localEndpoint) {
  if (localEndpoint.transport() != Transport::CudaIpc || remoteEndpoint.transport() != Transport::CudaIpc) {
    THROW(CONN, Error, ErrorCode::InternalError, "CudaIpc transport is required for CudaIpcConnection");
  }
  if (localEndpoint.device().type == DeviceType::GPU && localEndpoint.device().id < 0) {
    THROW(CONN, Error, ErrorCode::InternalError, "No GPU device ID provided for local endpoint");
  }
  if (remoteEndpoint.device().type == DeviceType::GPU && remoteEndpoint.device().id < 0) {
    THROW(CONN, Error, ErrorCode::InternalError, "No GPU device ID provided for remote endpoint");
  }
  int localDeviceId = localEndpoint.device().id;
  int remoteDeviceId = remoteEndpoint.device().id;
  if (localEndpoint.device().type != DeviceType::GPU && remoteEndpoint.device().type != DeviceType::GPU) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "CudaIpcConnection requires at least one GPU endpoint");
  } else if (localEndpoint.device().type == DeviceType::GPU && remoteEndpoint.device().type == DeviceType::GPU) {
    if (isSameProcess(localEndpoint, remoteEndpoint) && localDeviceId != remoteDeviceId) {
      // Connecting two GPUs in the same process - need to enable peer access explicitly
      CudaDeviceGuard deviceGuard(localDeviceId);
      auto ret = cudaDeviceEnablePeerAccess(remoteDeviceId, 0);
      if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
        MSCCLPP_CUDATHROW(ret);
      }
    }
  }
  int streamDeviceId = (localEndpoint.device().type == DeviceType::GPU) ? localDeviceId : remoteDeviceId;
  auto& ctxImpl = getImpl(*context);
#if defined(MSCCLPP_DEVICE_HIP)
  ctxImpl.ipcStreams_.emplace_back(std::make_shared<CudaIpcStream>(streamDeviceId));
#else   // !defined(MSCCLPP_DEVICE_HIP)
  if (ctxImpl.ipcStreams_.empty()) {
    ctxImpl.ipcStreams_.emplace_back(std::make_shared<CudaIpcStream>(streamDeviceId));
  }
#endif  // !defined(MSCCLPP_DEVICE_HIP)
  stream_ = ctxImpl.ipcStreams_.back();
}

Transport CudaIpcConnection::transport() const { return Transport::CudaIpc; }

Transport CudaIpcConnection::remoteTransport() const { return Transport::CudaIpc; }

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                              uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_WRITE_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  char* dstPtr = (char*)dst.data();
  char* srcPtr = (char*)src.data();

  stream_->memcpyD2D(dstPtr + dstOffset, srcPtr + srcOffset, size);

  INFO(CONN, "CudaIpcConnection write: from ", srcPtr + srcOffset, " to ", dstPtr + dstOffset, ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_WRITE_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void CudaIpcConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport());
  uint64_t oldValue = *src;
  *src = newValue;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.data()) + dstOffset);

  stream_->memcpyH2D(dstPtr + dstOffset, src, sizeof(uint64_t));

  INFO(CONN, "CudaIpcConnection atomic write: from ", src, " to ", dstPtr + dstOffset, ", ", oldValue, " -> ",
       newValue);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_UPDATE_AND_SYNC_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void CudaIpcConnection::flush(int64_t timeoutUsec) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  if (timeoutUsec >= 0) {
    INFO(CONN, "CudaIpcConnection flush: timeout is not supported, ignored");
  }

  stream_->sync();

  INFO(CONN, "CudaIpcConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_CUDA_IPC_FLUSH_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

// IBConnection

void IBConnection::recvThreadFunc() {
  // Set the CUDA device context for this thread
  if (localGpuDeviceId_ >= 0) {
    cudaError_t err = cudaSetDevice(localGpuDeviceId_);
    if (err != cudaSuccess) {
      WARN(NET, "IBConnection recvThreadFunc: cudaSetDevice(", localGpuDeviceId_,
           ") failed: ", cudaGetErrorString(err));
      return;
    }
    // Bind this thread to the NUMA node of the local GPU for optimal memory access
    int deviceNumaNode = getDeviceNumaNode(localGpuDeviceId_);
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
    }
  }

  uint32_t lastImmData = 0;
  uint64_t immHighBits = 0;
  uint64_t newValueHost = 0;

  auto qp = qp_.lock();
  if (!qp) return;

  while (!stopRecvThread_.load(std::memory_order_relaxed)) {
    int wcNum = qp->pollRecvCq();
    if (wcNum < 0) {
      recvThreadErrorMsg_ = "pollRecvCq failed";
      recvThreadError_.store(true, std::memory_order_release);
      WARN(NET, "IBConnection recvThreadFunc: ", recvThreadErrorMsg_);
      break;
    }

    for (int i = 0; i < wcNum; ++i) {
      int status = qp->getRecvWcStatus(i);
      if (status != static_cast<int>(WsStatus::Success)) {
        // A failed recv WC typically means the QP entered error state (e.g., WR Flushed Error).
        // All remaining WRs will also fail — no recovery without QP recreation. Exit the thread
        // and set the error flag so the main thread can detect it.
        recvThreadErrorMsg_ = std::string("recv work completion failed: ") + qp->getRecvWcStatusString(i);
        recvThreadError_.store(true, std::memory_order_release);
        WARN(NET, "IBConnection recvThreadFunc: ", recvThreadErrorMsg_);
        return;
      }

      // Read the lower 32 bits of the token from imm_data. Reconstruct the full 64-bit value
      // using wrap-around detection: tokens increase monotonically, so if the new lower 32 bits
      // are less than the previous value, the upper 32 bits must have incremented by 1.
      uint32_t immData = qp->getRecvWcImmData(i);
      if (immData < lastImmData) {
        immHighBits += (1ULL << 32);
      }
      lastImmData = immData;
      newValueHost = immHighBits | static_cast<uint64_t>(immData);

      // Forward the token to the semaphore's inbound token address via atomicStore
      // through the GDRCopy BAR1 mapping. The GPU reads with system-scope acquire.
      if (signalAddr_ != 0) {
        if (signalGdrMap_ && signalGdrMap_->valid()) {
          atomicStore(signalGdrMap_->hostPtr(), newValueHost, memoryOrderRelaxed);
        } else {
          // For HIP/ROCm.
          // NOTE: may need a fix in the future to ensure BAR1 mapping.
          *reinterpret_cast<volatile uint64_t*>(signalAddr_) = newValueHost;
        }
      }

      // Post another recv for future messages
      qp->stageRecv(/*wrId=*/0);
      qp->postRecv();
    }
  }
}

IBConnection::IBConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                           const Endpoint& remoteEndpoint)
    : BaseConnection(context, localEndpoint),
      transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      atomicSrc_(std::make_unique<uint64_t>(0)),
      ibNoAtomic_(getImpl(localEndpoint).ibNoAtomic_),
      gdrSignalForwarding_(false),
      stopRecvThread_(false),
      recvThreadError_(false),
      localGpuDeviceId_(localEndpoint.device().id),
      signalAddr_(0) {
  qp_ = getImpl(localEndpoint).ibQp_;
  qp_.lock()->rtr(getImpl(remoteEndpoint).ibQpInfo_);
  qp_.lock()->rts();
  atomicSrcMem_ = context->registerMemory(atomicSrc_.get(), sizeof(uint64_t), transport_);
  validateTransport(atomicSrcMem_, transport_);
  atomicSrcTransportInfo_ =
      std::get<detail::TransportInfo<IBTransportTag>>(getImpl(atomicSrcMem_).getTransportInfo(transport_).data);

  if (ibNoAtomic_) {
#if defined(MSCCLPP_USE_CUDA)
    // On CUDA, HostNoAtomic requires GDRCopy for CPU→GPU signal forwarding through BAR1.
    if (!gdrEnabled()) {
      THROW(CONN, Error, ErrorCode::InvalidUsage,
            "IB host-no-atomic mode on CUDA requires GDRCopy: ", gdrStatusMessage());
    }
    gdrSignalForwarding_ = true;
#endif  // defined(MSCCLPP_USE_CUDA)

    // On platforms with a CPU-GPU bridge that reorders posted writes (e.g., Grace/GB200
    // NVLink-C2C), HostNoAtomic requires Data Direct for correct memory ordering. Data Direct
    // routes NIC DMA through the PCIe Data Direct engine, bypassing the bridge. It is available
    // on Virtual Function (VF) devices. On platforms without such a bridge (x86, non-Grace
    // aarch64), HostNoAtomic works without Data Direct.
    //
    // We cannot reliably detect the bridge at compile time or runtime, so we emit a warning
    // when the device is not a VF. If data corruption occurs, switching to VF devices with
    // Data Direct or using IbMode::Host with RDMA atomics will resolve it.
    {
      IbCtx* ibCtx = getImpl(*context).getIbContext(transport_);
      if (!ibCtx->isVirtualFunction()) {
        WARN(CONN,
             "IB HostNoAtomic mode without a Virtual Function (VF) device may cause data corruption "
             "on platforms with a CPU-GPU bridge that reorders posted writes (e.g., Grace/GB200). "
             "Device ",
             ibCtx->getDevName(),
             " is not a VF. "
             "If you experience data corruption, use VF devices with Data Direct or IbMode::Host.");
      }
    }

    // Pre-post receive requests for incoming WRITE_WITH_IMM notifications.
    // The recv CQE guarantees the preceding data WRITE has been committed to GPU memory.
    auto qp = qp_.lock();
    int maxRecvWr = localEndpoint.config().ib.maxRecvWr;
    for (int i = 0; i < maxRecvWr; ++i) {
      qp->stageRecv(/*wrId=*/0);
    }
    qp->postRecv();
    // The recv thread is started later in startSignalForwarding() when the semaphore
    // provides the signal forwarding destination. This ensures the thread lifetime is
    // bounded by the GdrMap lifetime (created before start, destroyed after stop).
    INFO(CONN, "IBConnection via ", getIBDeviceName(transport_), " created with signal forwarding (HostNoAtomic) mode");
  } else {
    INFO(CONN, "IBConnection via ", getIBDeviceName(transport_), " created with atomic mode");
  }
}

IBConnection::~IBConnection() { stopSignalForwarding(); }

Transport IBConnection::transport() const { return transport_; }

Transport IBConnection::remoteTransport() const { return remoteTransport_; }

bool IBConnection::isSignalForwarding() const { return ibNoAtomic_; }

void IBConnection::startSignalForwarding(std::shared_ptr<uint64_t> mem) {
  // Set up the forwarding destination and GdrMap, then start the recv thread.
  // Order: set address → create GdrMap → start thread.
  signalAddr_ = reinterpret_cast<uint64_t>(mem.get());
  if (gdrSignalForwarding_) {
    signalGdrMap_ = std::make_unique<GdrMap>(std::move(mem), localGpuDeviceId_);
  }
  if (ibNoAtomic_) {
    stopRecvThread_.store(false, std::memory_order_relaxed);
    recvThread_ = std::thread([this]() { this->recvThreadFunc(); });
  }
  INFO(CONN, "IBConnection startSignalForwarding: ", (void*)signalAddr_);
}

void IBConnection::stopSignalForwarding() {
  // Stop the recv thread, then tear down GdrMap and address.
  // Order: stop thread → destroy GdrMap → clear address.
  if (ibNoAtomic_) {
    stopRecvThread_.store(true, std::memory_order_relaxed);
    if (recvThread_.joinable()) {
      recvThread_.join();
    }
  }
  if (gdrSignalForwarding_) {
    signalGdrMap_.reset();
  }
  signalAddr_ = 0;
  INFO(CONN, "IBConnection stopSignalForwarding");
}

void IBConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_WRITE_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  auto& dstData = std::get<detail::TransportInfo<IBTransportTag>>(dstTransportInfo.data);
  if (dstData.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "dst is local, which is not supported");
  }
  auto srcTransportInfo = getImpl(src).getTransportInfo(transport());
  auto& srcData = std::get<detail::TransportInfo<IBTransportTag>>(srcTransportInfo.data);
  if (!srcData.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "src is remote, which is not supported");
  }

  auto dstMrInfo = dstData.ibMrInfo;
  auto srcMr = srcData.ibMr;

  qp_.lock()->stageSendWrite(srcMr, dstMrInfo, (uint32_t)size, /*wrId=*/0, /*srcOffset=*/srcOffset,
                             /*dstOffset=*/dstOffset, /*signaled=*/true);

  qp_.lock()->postSend();
  INFO(CONN, "IBConnection write: from ", (uint8_t*)srcMr->getBuff() + srcOffset, " to ",
       (uint8_t*)dstMrInfo.addr + dstOffset, ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_WRITE_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void IBConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  validateTransport(dst, remoteTransport());
  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  auto& dstData = std::get<detail::TransportInfo<IBTransportTag>>(dstTransportInfo.data);
  if (dstData.ibLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "dst is local, which is not supported");
  }

  auto dstMrInfo = dstData.ibMrInfo;
  // assert that src is on host
  uint64_t oldValue = *src;
  *src = newValue;

  if (ibNoAtomic_) {
    // Signal forwarding: send a 0-byte RDMA WRITE_WITH_IMM with the lower 32 bits of the
    // token in imm_data. The receiver reconstructs the full 64-bit value using wrap-around
    // detection (tokens are monotonically increasing, so a decrease in the lower 32 bits
    // indicates the upper 32 bits incremented by 1).
    if (newValue <= oldValue) {
      WARN(CONN, "IBConnection signal forwarding: token is not monotonically increasing: ", oldValue, " -> ", newValue);
    } else if (newValue - oldValue >= (1ULL << 32)) {
      WARN(CONN,
           "IBConnection signal forwarding: token increment too large for 32-bit wrap-around detection: ", oldValue,
           " -> ", newValue, " (delta ", newValue - oldValue, " >= 2^32)");
    }
    unsigned int immData = static_cast<unsigned int>(newValue);
    qp_.lock()->stageSendWriteWithImm(nullptr, dstMrInfo,
                                      /*size=*/0, /*wrId=*/0,
                                      /*srcOffset=*/0, /*dstOffset=*/0,
                                      /*signaled=*/true, /*immData=*/immData);
    qp_.lock()->postSend();
    INFO(CONN, "IBConnection signal forwarding: value ", oldValue, " -> ", newValue);
  } else {
    qp_.lock()->stageSendAtomicAdd(atomicSrcTransportInfo_.ibMr, dstMrInfo, /*wrId=*/0, dstOffset, newValue - oldValue,
                                   /*signaled=*/true);
    qp_.lock()->postSend();
    INFO(CONN, "IBConnection atomic write: from ", src, " to ", (uint8_t*)dstMrInfo.addr + dstOffset, ", ", oldValue,
         " -> ", newValue);
  }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_UPDATE_AND_SYNC_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void IBConnection::flush(int64_t timeoutUsec) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_FLUSH_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  // Check if the recv thread has already reported an error (e.g., QP entered error state).
  if (recvThreadError_.load(std::memory_order_acquire)) {
    THROW(CONN, Error, ErrorCode::SystemError, "IBConnection recv thread failed: ", recvThreadErrorMsg_);
  }

  Timer timer;
  while (qp_.lock()->getNumSendCqItems()) {
    int wcNum = qp_.lock()->pollSendCq();
    if (wcNum < 0) {
      THROW(NET, IbError, errno, "pollSendCq failed");
    } else if (timeoutUsec >= 0) {
      auto elapsed = timer.elapsed();
      if (elapsed > timeoutUsec) {
        THROW(CONN, Error, ErrorCode::Timeout, "pollSendCq timed out: waited for ", elapsed / 1e6,
              " seconds. Expected ", qp_.lock()->getNumSendCqItems(), " signals");
      }
    }
    for (int i = 0; i < wcNum; ++i) {
      int status = qp_.lock()->getSendWcStatus(i);
      if (status != static_cast<int>(WsStatus::Success)) {
        THROW(NET, Error, ErrorCode::SystemError, "an IB work item failed: ", qp_.lock()->getSendWcStatusString(i));
      }
    }
  }
  INFO(CONN, "IBConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_IB_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_IB_FLUSH_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

// EthernetConnection

EthernetConnection::EthernetConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                                       const Endpoint& remoteEndpoint, uint64_t sendBufferSize, uint64_t recvBufferSize)
    : BaseConnection(context, localEndpoint),
      abortFlag_(0),
      sendBufferSize_(sendBufferSize),
      recvBufferSize_(recvBufferSize) {
  // Validating Transport Protocol
  if (localEndpoint.transport() != Transport::Ethernet || remoteEndpoint.transport() != Transport::Ethernet) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "Ethernet connection can only be made from Ethernet endpoints");
  }

  // Instanciating Buffers
  sendBuffer_.resize(sendBufferSize_);
  recvBuffer_.resize(recvBufferSize_);

  // Creating Thread to Accept the Connection
  auto parameter = getImpl(localEndpoint).socket_.get();
  std::thread t([this, parameter]() {
    recvSocket_ = std::make_unique<Socket>(nullptr, MSCCLPP_SOCKET_MAGIC, SocketTypeUnknown, abortFlag_);
    recvSocket_->accept(parameter);
  });

  // Starting Connection
  sendSocket_ = std::make_unique<Socket>(&(getImpl(remoteEndpoint).socketAddress_), MSCCLPP_SOCKET_MAGIC,
                                         SocketTypeBootstrap, abortFlag_);
  sendSocket_->connect();

  // Ensure the Connection was Established
  t.join();

  // Starting Thread to Receive Messages
  int deviceId = -1;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  threadRecvMessages_ = std::thread([deviceId, this]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(deviceId));
    this->recvMessages();
  });

  INFO(CONN, "Ethernet connection created");
}

EthernetConnection::~EthernetConnection() {
  sendSocket_->close();
  recvSocket_->close();
  threadRecvMessages_.join();
}

Transport EthernetConnection::transport() const { return Transport::Ethernet; }

Transport EthernetConnection::remoteTransport() const { return Transport::Ethernet; }

void EthernetConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                               uint64_t size) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_WRITE_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_WRITE_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  // Validating Transport Protocol
  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  // Initializing Variables
  char* srcPtr = reinterpret_cast<char*>(src.data()) + srcOffset / sizeof(char);
  char* dstPtr = reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset / sizeof(char);
  uint64_t sentDataSize = 0;
  uint64_t headerSize = 0;

  // Copying Meta Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr), sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&size);
  std::copy(sizeBytes, sizeBytes + sizeof(size), sendBuffer_.data() + headerSize / sizeof(char));
  headerSize += sizeof(size);

  // Getting Data From GPU and Sending Message
  while (sentDataSize < size) {
    uint64_t dataSize =
        std::min(sendBufferSize_ - headerSize / sizeof(char), (size - sentDataSize) / sizeof(char)) * sizeof(char);
    uint64_t messageSize = dataSize + headerSize;
    mscclpp::gpuMemcpy(sendBuffer_.data() + headerSize / sizeof(char), srcPtr + (sentDataSize / sizeof(char)), dataSize,
                       cudaMemcpyDeviceToHost);
    sendSocket_->send(sendBuffer_.data(), messageSize);
    sentDataSize += messageSize;
    headerSize = 0;
  }

  INFO(CONN, "EthernetConnection write: from ", srcPtr, " to ", dstPtr, ", size ", size);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_WRITE_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_WRITE_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  // Validating Transport Protocol
  validateTransport(dst, remoteTransport());

  // Initializing Variables
  uint64_t oldValue = *src;
  uint64_t* dstPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(dst.originalDataPtr()) + dstOffset);
  uint64_t dataSize = sizeof(uint64_t);
  uint64_t messageSize = 0;
  *src = newValue;

  // Copying Data to Send Buffer
  char* dstPtrBytes = reinterpret_cast<char*>(&dstPtr);
  std::copy(dstPtrBytes, dstPtrBytes + sizeof(dstPtr), sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dstPtr);
  char* sizeBytes = reinterpret_cast<char*>(&dataSize);
  std::copy(sizeBytes, sizeBytes + sizeof(dataSize), sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += sizeof(dataSize);
  char* dataBytes = reinterpret_cast<char*>(src);
  std::copy(dataBytes, dataBytes + dataSize, sendBuffer_.data() + messageSize / sizeof(char));
  messageSize += dataSize;

  // Sending Message
  sendSocket_->send(sendBuffer_.data(), messageSize);

  INFO(CONN, "EthernetConnection atomic write: from ", src, " to ", dstPtr + dstOffset, ", ", oldValue, " -> ",
       newValue);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_UPDATE_AND_SYNC_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::flush(int64_t) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_FLUSH_ENTRY)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_FLUSH_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif

  INFO(CONN, "EthernetConnection flushing connection");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_FLUSH_EXIT)
  NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_FLUSH_EXIT, 0, 0, *NpKit::GetCpuTimestamp(), 0);
#endif
}

void EthernetConnection::recvMessages() {
  // Declarating Variables
  char* ptr;
  uint64_t size;
  uint64_t recvSize;
  int closed = 0;
  bool received = true;

  // Receiving Messages Until Connection is Closed
  while (recvSocket_->getState() != SocketStateClosed) {
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_META_ENTRY)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_META_ENTRY, 0, 0, *NpKit::GetCpuTimestamp(), 1);
#endif

    // Receiving Data Address
    if (closed == 0) recvSocket_->recvUntilEnd(&ptr, sizeof(char*), &closed);
    received &= !closed;

    // Receiving data size
    if (closed == 0) recvSocket_->recvUntilEnd(&size, sizeof(uint64_t), &closed);
    received &= !closed;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_META_EXIT)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_META_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 1);
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_DATA_ENTRY)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_DATA_ENTRY, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 1);
#endif

    // Receiving Data and Copying Data yo GPU
    recvSize = 0;
    while (recvSize < size && closed == 0) {
      uint64_t messageSize = std::min(recvBufferSize_, (size - recvSize) / sizeof(char)) * sizeof(char);
      recvSocket_->recvUntilEnd(recvBuffer_.data(), messageSize, &closed);
      received &= !closed;

      if (received)
        mscclpp::gpuMemcpy(ptr + (recvSize / sizeof(char)), recvBuffer_.data(), messageSize, cudaMemcpyHostToDevice);
      recvSize += messageSize;
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_CONN_ETH_RECV_DATA_EXIT)
    NpKit::CollectCpuEvent(NPKIT_EVENT_CONN_ETH_RECV_DATA_EXIT, uint32_t(size), 0, *NpKit::GetCpuTimestamp(), 1);
#endif
  }
}


namespace {

#if defined(MSCCLPP_USE_OFI)
constexpr uint32_t kOfiEndpointFlagWriteData = 1u << 0;

bool envEnabled(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) {
    return false;
  }
  return (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0 ||
          std::strcmp(v, "on") == 0 || std::strcmp(v, "ON") == 0);
}

void checkOfiConn(int rc, char const* what) {
  if (rc < 0) {
    THROW(CONN, Error, ErrorCode::SystemError, what, " failed: ", fi_strerror(-rc), " (rc=", rc, ")");
  }
}

struct OfiConsumeGuard {
  explicit OfiConsumeGuard(std::atomic<bool>& consumed) : consumed_(&consumed) {}

  void dismiss() noexcept { consumed_ = nullptr; }

  ~OfiConsumeGuard() {
    if (consumed_ != nullptr) {
      consumed_->store(false, std::memory_order_release);
    }
  }

 private:
  std::atomic<bool>* consumed_;
};
#endif  // defined(MSCCLPP_USE_OFI)

}  // namespace

struct OfiConnection::Impl {
  Impl() = default;

  std::unique_ptr<OfiEndpointResources> resources;  // owned by the connection after consume
#if defined(MSCCLPP_USE_OFI)
  fi_addr_t peerAddr = FI_ADDR_UNSPEC;
#endif  // defined(MSCCLPP_USE_OFI)
  uint64_t rxCqCompletionsSeen = 0;
  uint64_t cntrCompletionsSeen = 0;
  uint64_t remoteUpdateDstAddr = 0;

  std::unique_ptr<uint64_t> updateScratch;
  unique_memregion updateScratchMr;
};

OfiConnection::OfiConnection(std::shared_ptr<Context> context, Endpoint const& localEndpoint,
                             Endpoint const& remoteEndpoint)
    : BaseConnection(context, localEndpoint) {
  if (localEndpoint.transport() != Transport::Ofi || remoteEndpoint.transport() != Transport::Ofi) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection requires Transport::Ofi endpoints");
  }

#if defined(MSCCLPP_USE_OFI)
  Endpoint& mutableLocalEndpoint = localEndpoint_;
  auto& localImpl = getImpl(mutableLocalEndpoint);
  auto const& remoteImpl = getImpl(remoteEndpoint);

  if (!localImpl.ofiResources_) {
    THROW(CONN, Error, ErrorCode::InternalError,
          "Local OFI endpoint is missing runtime OFI resources");
  }
  if (remoteImpl.ofiWireInfo_.addr.empty()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "Remote OFI endpoint has no serialized address");
  }

  bool expected = false;
  if (!localImpl.ofiConsumed_.compare_exchange_strong(expected, true,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "Local OFI endpoint has already been consumed by another OfiConnection");
  }
  OfiConsumeGuard consumeGuard(localImpl.ofiConsumed_);

  impl_ = std::make_unique<Impl>();

  // While construction is in progress, the endpoint still temporarily owns the resources.
  OfiEndpointResources* localResources = localImpl.ofiResources_.get();
  auto controller = localResources->controller();

  impl_->peerAddr = static_cast<fi_addr_t>(remoteImpl.ofiWireInfo_.rank);
  auto worldSize = static_cast<fi_addr_t>(localResources->communicator()->size());
  if (impl_->peerAddr >= worldSize) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "OFI peer fi_address out of range: peerAddr=", static_cast<uint64_t>(impl_->peerAddr),
          " worldSize=", static_cast<uint64_t>(worldSize));
  }
  INFO(CONN, "Using peer address index from wire rank: ", static_cast<uint64_t>(impl_->peerAddr));

  // Ownership transfer:
  // once the endpoint is consumed successfully, the connection becomes the
  // exclusive owner of the OFI EP/AV/CQ/progress resources.
  impl_->resources = std::move(localImpl.ofiResources_);

  const bool localWriteData = localResources->supportsWriteData();
  const bool remoteWriteData = localWriteData; // (remoteImpl.ofiWireInfo_.flags & kOfiEndpointFlagWriteData) != 0;
  const std::string& ofiProvider = localEndpoint.config().ofi.provider;
  impl_->resources->communicator()->configure_runtime_options(ofiProvider, localWriteData, remoteWriteData,
                                                              envEnabled("MSCCLPP_OFI_INJECT_WRITEDATA"));
  
  impl_->updateScratch = std::make_unique<uint64_t>(0);

  auto attr = classifyOfiMemory(impl_->updateScratch.get());
  auto region = libfatbat::make_region(controller, impl_->updateScratch.get(), sizeof(uint64_t), attr.device);
  impl_->updateScratchMr = unique_memregion(new libfatbat::memory_region(region), memregion_deleter());

  INFO(CONN, "OfiConnection created: local EP ", impl_->resources->ep(),
       ", peerAddr=", static_cast<uint64_t>(impl_->peerAddr),
       ", remoteAddrBytes=", remoteImpl.ofiWireInfo_.addr.size(),
       ", localWriteData=", localWriteData,
       ", remoteWriteData=", remoteWriteData,
       ", provider=", ofiProvider,
      ", useRelativeRemoteAddr=", controller->use_relative_remote_addr(),
      ", useTxCounterFlush=", impl_->resources->communicator()->use_tx_counter_flush(),
      ", useWriteDataSignal=", impl_->resources->communicator()->use_write_data_signal(),
      ", preferInjectWriteData=", impl_->resources->communicator()->prefer_inject_writedata());

  impl_->resources->controller()->set_remote_cq_data_callback(
      [this](uint64_t value) { this->handleRemoteCqData(value); });

  consumeGuard.dismiss();
#else
  (void)remoteEndpoint;
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

OfiConnection::~OfiConnection() {
#if defined(MSCCLPP_USE_OFI)
  if (impl_ && impl_->resources) {
    impl_->resources->controller()->clear_remote_cq_data_callback();
  }
#endif
}

void OfiConnection::handleRemoteCqData(uint64_t value) {
#if defined(MSCCLPP_USE_OFI)
  if (!impl_) {
    return;
  }

  auto* comm = impl_->resources->communicator();

  ++impl_->rxCqCompletionsSeen;
  value = comm->normalize_remote_cq_data(value);

  if (impl_->remoteUpdateDstAddr == 0) {
    return;
  }

  auto* dstPtr = reinterpret_cast<uint64_t*>(impl_->remoteUpdateDstAddr);
  int dstGpuId = detail::gpuIdFromAddress(dstPtr);
  int currentDevice = -1;
  (void)cudaGetDevice(&currentDevice);
  if (localEndpoint_.device().type == DeviceType::GPU && dstGpuId >= 0) {
    CudaDeviceGuard deviceGuard(localEndpoint_.device().id);
    MSCCLPP_CUTHROW(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dstPtr), &value, sizeof(value)));
  } else {
    atomicStore(dstPtr, value, memoryOrderRelease);
  }

  DEBUG(CONN, "OfiConnection inbound signal: value=", value,
        " dst=", dstPtr,
        " dstGpuId=", dstGpuId,
        " currentDevice=", currentDevice,
        " rxCqCompletionsSeen=", impl_->rxCqCompletionsSeen);
#endif
}

Transport OfiConnection::transport() const { return Transport::Ofi; }

Transport OfiConnection::remoteTransport() const { return Transport::Ofi; }

bool OfiConnection::isSignalForwarding() const {
  INFO(CONN, "OfiConnection isSignalForwarding: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  return impl_ && impl_->resources && impl_->resources->communicator()->use_write_data_signal();
#else
  return false;
#endif
}

void OfiConnection::startSignalForwarding(std::shared_ptr<uint64_t> mem) {
  INFO(CONN, "OfiConnection startSignalForwarding: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }
  if (!impl_->resources->communicator()->use_write_data_signal()) {
    return;
  }
  impl_->remoteUpdateDstAddr = reinterpret_cast<uint64_t>(mem.get());
  INFO(CONN, "OfiConnection startSignalForwarding: ", reinterpret_cast<void*>(impl_->remoteUpdateDstAddr));
#else
  (void)mem;
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

void OfiConnection::stopSignalForwarding() {
  INFO(CONN, "OfiConnection stopSignalForwarding: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_) {
    return;
  }
  impl_->remoteUpdateDstAddr = 0;
  INFO(CONN, "OfiConnection stopSignalForwarding");
#endif
}

void OfiConnection::progress() {
  // INFO(CONN, "OfiConnection progress: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  (void)progressInboundSignalsOnce();
#endif
}

void OfiConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                          uint64_t size) {

  INFO(CONN, "OfiConnection write: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }
  validateTransport(dst, remoteTransport(), dstOffset, size);
  validateTransport(src, transport(), srcOffset, size);

  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  auto& dstData = std::get<TransportInfoType<Transport::Ofi>>(dstTransportInfo.data);
  if (dstData.ofiLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "dst is local, which is not supported");
  }

  auto srcTransportInfo = getImpl(src).getTransportInfo(transport());
  auto& srcData = std::get<TransportInfoType<Transport::Ofi>>(srcTransportInfo.data);
  if (!srcData.ofiLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "src is remote, which is not supported");
  }
  if (srcData.ofiMr == nullptr) {
    THROW(CONN, Error, ErrorCode::InternalError, "Local OFI source memory is missing an OFI MR");
  }

  const auto srcAttr = classifyOfiMemory(getImpl(src).data);
  if (srcAttr.iface != libfatbat::mem_Iface::System && !impl_->resources->controller()->supports_hmem()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "OFI provider does not support FI_HMEM; GPU source memory is unsupported in OfiConnection::write");
  }

  const auto dstAttr = classifyOfiMemory(getImpl(dst).data);
  if (dstAttr.iface != libfatbat::mem_Iface::System && !impl_->resources->controller()->supports_hmem()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "OFI provider does not support FI_HMEM; GPU destination memory is unsupported in OfiConnection::write");
  }

  auto localBuf = static_cast<void*>(static_cast<uint8_t*>(getImpl(src).data) + srcOffset);
  // CXI path uses MR-relative remote addresses when FI_MR_VIRT_ADDR is not requested.
  auto* comm = impl_->resources->communicator();
  auto remoteAddr = comm->remote_rma_addr_value(dstData.ofiMrInfo.addr, dstOffset);

  comm->write_remote_raw(localBuf, static_cast<size_t>(size), srcData.ofiMr->get_local_key(), impl_->peerAddr,
                         remoteAddr, dstData.ofiMrInfo.rkey, nullptr, []() {});

  DEBUG(CONN, "OfiConnection write: local=", localBuf,
       " remote=", reinterpret_cast<void*>(remoteAddr),
      " remoteBase=", reinterpret_cast<void*>(dstData.ofiMrInfo.addr),
      " remoteOffset=", dstOffset,
      " useRelativeRemoteAddr=", impl_->resources->controller()->use_relative_remote_addr(),
       " size=", size,
       " rkey=", dstData.ofiMrInfo.rkey,
       " postedWrites=", comm->posted_writes(),
       " writesPostedNoCq=", comm->writes_posted_no_cq(),
      " useWriteDataSignal=", comm->use_write_data_signal());
#else
  (void)dst;
  (void)dstOffset;
  (void)src;
  (void)srcOffset;
  (void)size;
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

void OfiConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) {
  INFO(CONN, "OfiConnection updateAndSync: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }
  if (src == nullptr) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "src must not be null");
  }
  validateTransport(dst, remoteTransport(), dstOffset, sizeof(uint64_t));

  auto dstTransportInfo = getImpl(dst).getTransportInfo(remoteTransport());
  auto& dstData = std::get<TransportInfoType<Transport::Ofi>>(dstTransportInfo.data);
  if (dstData.ofiLocal) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "dst is local, which is not supported");
  }

  uint64_t oldValue = *src;
  *src = newValue;
  *impl_->updateScratch = newValue;
  auto* comm = impl_->resources->communicator();
  auto remoteAddr = comm->remote_rma_addr_value(dstData.ofiMrInfo.addr, dstOffset);

    DEBUG(CONN, "OfiConnection updateAndSync state: useWriteDataSignal=", comm->use_write_data_signal(),
      " preferInjectWriteData=", comm->prefer_inject_writedata(),
      " disableInjectWriteData=", comm->disable_inject_writedata(),
      " remoteUpdateDstAddr=", reinterpret_cast<void*>(impl_->remoteUpdateDstAddr),
      " oldValue=", oldValue,
      " newValue=", newValue);

  flush(-1);

  if (comm->use_write_data_signal()) {
    bool postedWriteData = false;

    DEBUG(CONN, "OfiConnection updateAndSync: posting signal update to peerAddr=", static_cast<uint64_t>(impl_->peerAddr),
          " remote=", reinterpret_cast<void*>(remoteAddr),
          " remoteBase=", reinterpret_cast<void*>(dstData.ofiMrInfo.addr),
          " remoteOffset=", dstOffset,
          " useRelativeRemoteAddr=", impl_->resources->controller()->use_relative_remote_addr(),
          " rkey=", dstData.ofiMrInfo.rkey,
          " value=", newValue);

    int signalRc = 0;
    auto signalResult = comm->post_update_signal(
      impl_->updateScratch.get(), sizeof(uint64_t), impl_->updateScratchMr->get_local_key(),
      impl_->peerAddr, remoteAddr, dstData.ofiMrInfo.rkey, newValue,
        /*fail_if_unsupported=*/impl_->remoteUpdateDstAddr != 0,
        [&postedWriteData]() { postedWriteData = true; }, &signalRc);

    if (signalResult == ofi_communicator::update_signal_result::error) {
      if ((signalRc == -FI_EOPNOTSUPP || signalRc == -FI_ENOSYS || signalRc == -FI_EINVAL) &&
          impl_->remoteUpdateDstAddr != 0) {
        THROW(CONN, Error, ErrorCode::SystemError,
              "fi_writedata unsupported after OFI signal forwarding has been initialized; "
              "cannot safely fall back to fi_write without risking token forwarding mismatch/hang. "
              "Disable FI_CXI_ENABLE_WRITEDATA or ensure provider/runtime supports fi_writedata. rc=",
              signalRc);
      }
      checkOfiConn(signalRc, "ofi_communicator::post_update_signal(updateAndSync)");
    }

    if (signalResult == ofi_communicator::update_signal_result::posted) {
      DEBUG(CONN, "OfiConnection XXXX updateAndSync(writedata): value ", oldValue, " -> ", newValue,
            " remote=", reinterpret_cast<void*>(remoteAddr),
            " remoteBase=", reinterpret_cast<void*>(dstData.ofiMrInfo.addr),
            " remoteOffset=", dstOffset,
            " useRelativeRemoteAddr=", impl_->resources->controller()->use_relative_remote_addr(),
            " rkey=", dstData.ofiMrInfo.rkey,
            " postedWrites=", comm->posted_writes(),
            " writesPostedNoCq=", comm->writes_posted_no_cq(),
            " cntrCompletionsSeen=", impl_->cntrCompletionsSeen);

      try {
        flush(5 * 1000 * 1000);
        return;
      } catch (const Error& e) {
        if (postedWriteData) {
          comm->rollback_write_posted_no_cq();
        }
        WARN(CONN,
             "Writedata path failed in updateAndSync, disabling useWriteDataSignal and falling back to fi_write. Reason: ",
             e.what());
        comm->disable_write_data_signal();
      }
    } else if (signalResult == ofi_communicator::update_signal_result::fallback_to_write) {
      WARN(CONN,
           "fi_writedata unsupported while OFI signal forwarding is enabled; disabling writedata signaling and falling back to fi_write.");
    }
  }

  if (!comm->use_tx_counter_flush()) {
    int rc = comm->inject_write_raw(impl_->updateScratch.get(), sizeof(uint64_t), impl_->peerAddr, remoteAddr,
                    dstData.ofiMrInfo.rkey, []() {});
    if (rc == 0) {
      DEBUG(CONN, "OfiConnection ZZZZ updateAndSync(inject-write): value ", oldValue, " -> ", newValue,
            " remote=", reinterpret_cast<void*>(remoteAddr),
            " remoteBase=", reinterpret_cast<void*>(dstData.ofiMrInfo.addr),
            " remoteOffset=", dstOffset,
            " useRelativeRemoteAddr=", impl_->resources->controller()->use_relative_remote_addr(),
            " rkey=", dstData.ofiMrInfo.rkey,
            " useTxCounterFlush=", comm->use_tx_counter_flush());
      return;
    }
    if (rc != -FI_EOPNOTSUPP && rc != -FI_ENOSYS && rc != -FI_EINVAL) {
      checkOfiConn(rc, "fi_inject_write(updateAndSync)");
    }
  }

  comm->write_remote_raw(impl_->updateScratch.get(), sizeof(uint64_t), impl_->updateScratchMr->get_local_key(),
                         impl_->peerAddr, remoteAddr, dstData.ofiMrInfo.rkey, nullptr, []() {});

  DEBUG(CONN, "OfiConnection YYYY updateAndSync(fallback-write): value ", oldValue, " -> ", newValue,
       " remote=", reinterpret_cast<void*>(remoteAddr),
       " remoteBase=", reinterpret_cast<void*>(dstData.ofiMrInfo.addr),
       " remoteOffset=", dstOffset,
       " useRelativeRemoteAddr=", impl_->resources->controller()->use_relative_remote_addr(),
       " rkey=", dstData.ofiMrInfo.rkey,
       " postedWrites=", comm->posted_writes(),
      " writesPostedNoCq=", comm->writes_posted_no_cq(),
       " cntrCompletionsSeen=", impl_->cntrCompletionsSeen);

  if (comm->use_tx_counter_flush()) {
    flush(5 * 1000 * 1000);
  }
#else
  (void)dst;
  (void)dstOffset;
  (void)src;
  (void)newValue;
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

void OfiConnection::flush(int64_t timeoutUsec) {
  INFO(CONN, "OfiConnection flush: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }

    auto* comm = impl_->resources->communicator();
    const uint64_t target = comm->posted_writes();
  DEBUG(CONN, "OfiConnection::flush enter: target=", target,
      " postedWrites=", comm->posted_writes(),
      " writesPostedNoCq=", comm->writes_posted_no_cq(),
       " cntrCompletionsSeen=", impl_->cntrCompletionsSeen,
     " remoteSignalsSeen=", comm->inbound_signals_seen(),
       " timeoutUsec=", timeoutUsec);
  if (target == 0) {
    return;
  }

  try {
    comm->wait_for_write_completions(timeoutUsec);
  } catch (std::runtime_error const&) {
    THROW(CONN, Error, ErrorCode::Aborted,
          "OfiConnection::flush timed out waiting for write counter");
  }

  impl_->cntrCompletionsSeen = impl_->resources->controller()->get_tx_counter_value();

  DEBUG(CONN, "OfiConnection::flush: postedWrites=", comm->posted_writes(),
      " writesPostedNoCq=", comm->writes_posted_no_cq(),
       " cntrCompletionsSeen=", impl_->cntrCompletionsSeen,
      " remoteSignalsSeen=", comm->inbound_signals_seen());
#else
  (void)timeoutUsec;
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

bool OfiConnection::progressInboundSignalsOnce() {
  // INFO(CONN, "OfiConnection progressInboundSignalsOnce: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }
  auto* comm = impl_->resources->communicator();
  auto before = comm->inbound_signals_seen();
  impl_->resources->communicator()->progress();
  auto after = comm->inbound_signals_seen();
  return after > before;
#else
  return false;
#endif
}

bool OfiConnection::progressCompletionsOnce() {
  // INFO(CONN, "OfiConnection progressCompletionsOnce: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }

  auto beforeWrites = static_cast<uint64_t>(impl_->resources->controller()->writes_complete_);
  auto beforeRecvs = static_cast<uint64_t>(impl_->resources->controller()->recvs_complete_);
  impl_->resources->communicator()->progress();
  auto afterWrites = static_cast<uint64_t>(impl_->resources->controller()->writes_complete_);
  auto afterRecvs = static_cast<uint64_t>(impl_->resources->controller()->recvs_complete_);
  return (afterWrites > beforeWrites) || (afterRecvs > beforeRecvs);
#else
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

void OfiConnection::waitForCompletions(int64_t timeoutUsec, void* targetContext, bool drainAll) {
  INFO(CONN, "OfiConnection waitForCompletions: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }

  auto* target = static_cast<std::atomic<bool>*>(targetContext);

  auto const deadline =
      (timeoutUsec < 0)
          ? std::chrono::steady_clock::time_point::max()
          : std::chrono::steady_clock::now() + std::chrono::microseconds(timeoutUsec);

  auto done = [&]() -> bool {
    if (drainAll) {
      return static_cast<uint64_t>(impl_->resources->controller()->get_tx_counter_value()) >=
             impl_->resources->communicator()->posted_writes();
    }
    return target != nullptr && target->load(std::memory_order_acquire);
  };

  while (!done()) {
    if (progressCompletionsOnce()) {
      continue;
    }

    if (timeoutUsec >= 0 && std::chrono::steady_clock::now() >= deadline) {
      if (drainAll) {
        THROW(CONN, Error, ErrorCode::Aborted,
              "OfiConnection::flush timed out with pending writes");
      } else {
        THROW(CONN, Error, ErrorCode::Aborted,
              "OfiConnection wait for targeted completion timed out");
      }
    }

    std::this_thread::yield();
  }
#else
  (void)timeoutUsec;
  (void)targetContext;
  (void)drainAll;
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

unique_memregion OfiConnection::registerOfiMr(void* data, size_t size) const {
  INFO(CONN, "OfiConnection registerOfiMr: ", "peerAddr=", static_cast<uint64_t>(impl_->peerAddr));
#if defined(MSCCLPP_USE_OFI)
  if (!impl_ || !impl_->resources) {
    THROW(CONN, Error, ErrorCode::InternalError, "OfiConnection is not initialized");
  }
  auto memAttr = classifyOfiMemory(data);
  if (memAttr.iface != libfatbat::mem_Iface::System && !impl_->resources->controller()->supports_hmem()) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "OFI provider does not support FI_HMEM; cannot register GPU memory for OFI transport");
  }

  auto reg = libfatbat::make_region(impl_->resources->controller(), data, size, memAttr.device);
  return unique_memregion(new libfatbat::memory_region(reg), memregion_deleter());
#else
  (void)data;
  (void)size;
  THROW(CONN, Error, ErrorCode::InvalidUsage,
        "OFI transport requested but MSCCLPP was built without OFI support");
#endif
}

}  // namespace mscclpp
