#include <mscclpp/gpu_utils.hpp>

#include "ofi.hpp"

#if defined(MSCCLPP_USE_OFI)
#include "ofi_wrapper.hpp"
#endif  // defined(MSCCLPP_USE_OFI)

#include <cstring>
#include <mscclpp/errors.hpp>
#include <mutex>

#include "logger.hpp"

namespace mscclpp {


namespace {

#if defined(MSCCLPP_USE_OFI)

constexpr std::uint32_t kOfiEndpointFlagWriteData = 1u << 0;

constexpr uint64_t kOfiBindFlagsCq = FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION;
constexpr uint64_t kOfiBindFlagsCntr = FI_WRITE | FI_TRANSMIT;

[[noreturn]] void throwOfiError(char const* what, int rc) {
  THROW(NET, Error, ErrorCode::SystemError, what, " failed: ", fi_strerror(-rc), " (rc=", rc, ")");
}

void checkOfi(int rc, char const* what) {
  if (rc != 0) {
    throwOfiError(what, rc);
  }
}

void logOfiEndpointConfigOnce(fi_info const* baseInfo, fi_info const* epInfo, size_t queueSize,
                              size_t cqSize, uint64_t cqBindFlags, uint64_t cntrBindFlags) {
  static std::once_flag once;
  std::call_once(once, [&]() {
    INFO(NET,
         "OFI endpoint bring-up: base tx_attr.size=", baseInfo->tx_attr ? baseInfo->tx_attr->size : 0,
         " base rx_attr.size=", baseInfo->rx_attr ? baseInfo->rx_attr->size : 0,
         " base tx_attr.op_flags=", baseInfo->tx_attr ? static_cast<unsigned long long>(baseInfo->tx_attr->op_flags) : 0ull,
         " ep tx_attr.size=", epInfo->tx_attr ? epInfo->tx_attr->size : 0,
         " ep rx_attr.size=", epInfo->rx_attr ? epInfo->rx_attr->size : 0,
         " ep tx_attr.op_flags=", epInfo->tx_attr ? static_cast<unsigned long long>(epInfo->tx_attr->op_flags) : 0ull,
         " qsz=", queueSize,
         " cq.size=", cqSize,
         " cq_bind_flags=", static_cast<unsigned long long>(cqBindFlags),
         " cntr_bind_flags=", static_cast<unsigned long long>(cntrBindFlags));
  });
}

#endif  // defined(MSCCLPP_USE_OFI)

}  // namespace

OfiMemoryAttr classifyOfiMemory(void* data) {
  OfiMemoryAttr attr{
    /*.iface = */OfiHmemIface::System,
    /*.device = */-1,
    /*.deviceOnly = */false
  };
  auto dev = detail::gpuIdFromAddress(data);
  if (dev >=0) {
    attr.device = dev;
#if defined(MSCCLPP_USE_ROCM)
    attr.iface = OfiHmemIface::Rocr;
#else
    attr.iface = OfiHmemIface::Cuda;
#endif
    attr.deviceOnly = true;
  }
  return attr;
}


//OfiMr::OfiMr(fid_domain* domain, void* data, size_t size) {
//#if defined(MSCCLPP_USE_OFI)
//  std::cout << "        OfiMr: registering memory for address " << data << " with size " << size << " using  fi_mr_reg" << std::endl;
//  int rc = fi_mr_reg(domain,
//                     data,
//                     size,
//                     FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
//                     0,
//                     0,
//                     0,
//                     &mr_,
//                     nullptr);
//  checkOfi(rc, "fi_mr_reg");
//
//  info_.addr = reinterpret_cast<uint64_t>(data);
//  info_.rkey = fi_mr_key(mr_);
//  info_.size = static_cast<uint64_t>(size);
//  std::cout << "        OfiMr: registered memory with addr " << info_.addr << " and rkey " << info_.rkey << std::endl;
//#else
//  (void)domain;
//  (void)data;
//  (void)size;
//#endif
//}
OfiMr::OfiMr(OfiEndpointResources& epRes, void* data, size_t size, OfiMemoryAttr const& memAttr) {
#if defined(MSCCLPP_USE_OFI)
  DEBUG(NET, "OfiMr: registering memory for address ", data, " size=", size);
  auto& ctx = epRes.ctx();
  uint64_t mrMode = ctx.mrMode();

  int rc = 0;

  bool needRegAttr =
      (memAttr.iface != OfiHmemIface::System) &&
      (mrMode & FI_MR_HMEM);

  if (needRegAttr) {
    iovec iov = {};
    iov.iov_base = data;
    iov.iov_len = size;

    fi_mr_attr attr = {};
    attr.mr_iov = &iov;
    attr.iov_count = 1;
    //attr.access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    attr.access = FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE;
    attr.offset = 0;
    attr.requested_key = 0;
    attr.context = nullptr;
    attr.auth_key_size = 0;
    attr.auth_key = nullptr;
    attr.hmem_data = nullptr;
    attr.page_size = 0;
    attr.base_mr = nullptr;
    attr.sub_mr_cnt = 0;

    switch (memAttr.iface) {
      case OfiHmemIface::Cuda:
        attr.iface = FI_HMEM_CUDA;
        attr.device.cuda = memAttr.device;
        break;
      case OfiHmemIface::Rocr:
        attr.iface = FI_HMEM_ROCR;
        attr.device.rocr = memAttr.device;
        break;
      case OfiHmemIface::System:
      default:
        attr.iface = FI_HMEM_SYSTEM;
        break;
    }

    uint64_t flags = 0;
    if (memAttr.deviceOnly) {
      flags |= FI_HMEM_DEVICE_ONLY;
    }

    rc = fi_mr_regattr(ctx.domain(), &attr, flags, &mr_);
    checkOfi(rc, "fi_mr_regattr");
  } else {
    rc = fi_mr_reg(ctx.domain(),
                   data,
                   size,
                   //FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
                   FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE,
                   0,
                   0,
                   0,
                   &mr_,
                   nullptr);
    checkOfi(rc, "fi_mr_reg");
  }

  if (mrMode & FI_MR_ENDPOINT) {
    rc = fi_mr_bind(mr_, &epRes.ep()->fid, 0);
    checkOfi(rc, "fi_mr_bind");
    rc = fi_mr_enable(mr_);
    checkOfi(rc, "fi_mr_enable");
  }

  const bool useVirtualAddress = (mrMode & FI_MR_VIRT_ADDR) != 0;
  info_.addr = useVirtualAddress ? reinterpret_cast<uint64_t>(data) : 0;
  info_.rkey = fi_mr_key(mr_);
  info_.size = static_cast<uint64_t>(size);
  INFO(NET, "OfiMr: registered addr=", info_.addr, " rkey=", info_.rkey, " size=", info_.size,
       " mr_mode=", mrMode);
#else
  (void)epRes;
  (void)data;
  (void)size;
  (void)memAttr;
#endif
}

void OfiMr::close() noexcept {
#if defined(MSCCLPP_USE_OFI)
  if (mr_ != nullptr) {
    fi_close(&mr_->fid);
    mr_ = nullptr;
  }
#endif
}

void* OfiMr::desc() const {
#if defined(MSCCLPP_USE_OFI)
  return fi_mr_desc(mr_);
#else
  return nullptr;
#endif
}

OfiCtx::OfiCtx(EndpointConfig::Ofi const& config) {
#if defined(MSCCLPP_USE_OFI)
  fi_info* hints = fi_allocinfo();
  if (hints == nullptr) {
    THROW(NET, Error, ErrorCode::SystemError, "fi_allocinfo failed");
  }
  DEBUG(NET, "OfiCtx: allocated hints for provider ", config.provider, " domain=", config.domain);

  try {
    hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    //hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->caps = FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    //hints->domain_attr->threading = FI_THREAD_DOMAIN;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    // Use provider defaults for progress mode.

    if (!config.provider.empty()) {
      hints->fabric_attr->prov_name = strdup(config.provider.c_str());
      if (hints->fabric_attr->prov_name == nullptr) {
        fi_freeinfo(hints);
        THROW(NET, Error, ErrorCode::SystemError, "strdup failed for OFI provider");
      }
    }
    if (!config.domain.empty()) {
      hints->domain_attr->name = strdup(config.domain.c_str());
      if (hints->domain_attr->name == nullptr) {
        fi_freeinfo(hints);
        THROW(NET, Error, ErrorCode::SystemError, "strdup failed for OFI domain");
      }
    }

    DEBUG(NET, "OfiCtx: attempting fi_getinfo for provider ", config.provider, " domain=", config.domain);
    // Keep this at a generic version for endpoint/domain bring-up.
    int rc = fi_getinfo(FI_VERSION(2, 5), nullptr, nullptr, 0, hints, &info_);
    fi_freeinfo(hints);
    hints = nullptr;
    checkOfi(rc, "fi_getinfo(OfiCtx)");
    INFO(NET, "OfiCtx: fi_getinfo succeeded for provider ", config.provider, " domain=", config.domain,
         " mr_mode=", info_->domain_attr ? info_->domain_attr->mr_mode : 0,
         " caps=", info_->caps);

    caps_ = info_->caps;
    mrMode_ = info_->domain_attr->mr_mode;

    rc = fi_fabric(info_->fabric_attr, &fabric_, nullptr);
    checkOfi(rc, "fi_fabric(OfiCtx)");
    DEBUG(NET, "OfiCtx: fi_fabric succeeded for provider ", config.provider, " domain=", config.domain);

    rc = fi_domain(fabric_, info_, &domain_, nullptr);
    checkOfi(rc, "fi_domain(OfiCtx)");
    DEBUG(NET, "OfiCtx: fi_domain succeeded for provider ", config.provider, " domain=", config.domain);
  } catch (...) {
    if (hints != nullptr) {
      fi_freeinfo(hints);
    }
    WARN(NET, "OfiCtx: exception during initialization, closing resources for provider ", config.provider,
         " domain=", config.domain);
    closeAll();
    throw;
  }
#endif  // defined(MSCCLPP_USE_OFI)
}

//std::unique_ptr<const OfiMr> OfiCtx::registerMr(void* data, size_t size) const {
//  std::cout << "      OfiCtx::registerMr for address " << data << " with size " << size << " using provider " << info_->fabric_attr->prov_name
//            << " and domain " << info_->domain_attr->name << std::endl;
//  return std::make_unique<const OfiMr>(domain_, data, size);
//}

void OfiCtx::closeAll() noexcept {
#if defined(MSCCLPP_USE_OFI)
  if (domain_ != nullptr) {
    fi_close(&domain_->fid);
    domain_ = nullptr;
  }
  if (fabric_ != nullptr) {
    fi_close(&fabric_->fid);
    fabric_ = nullptr;
  }
  if (info_ != nullptr) {
    fi_freeinfo(info_);
    info_ = nullptr;
  }
#endif  // defined(MSCCLPP_USE_OFI)
}

OfiEndpointResources::OfiEndpointResources(OfiCtx& ctx, EndpointConfig const& config) : ctx_(&ctx) {
#if defined(MSCCLPP_USE_OFI)
  if (config.transport != Transport::Ofi) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "OfiEndpointResources requires Transport::Ofi");
  }
  fi_info* epInfo = fi_dupinfo(ctx.info());
  if (epInfo == nullptr) {
    THROW(NET, Error, ErrorCode::SystemError, "fi_dupinfo failed");
  }

  try {
    const size_t qsz =
        (config.maxWriteQueueSize > 0) ? static_cast<size_t>(config.maxWriteQueueSize) : 4096;

    // Important: size endpoint queues, not just the CQ.
    epInfo->tx_attr->size = qsz;
    epInfo->rx_attr->size = qsz;

    //epInfo->tx_attr->op_flags |= (FI_INJECT_COMPLETE | FI_COMPLETION);
    epInfo->tx_attr->op_flags = 0;

    DEBUG(NET, "OfiEndpointResources: creating AV for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name);
    fi_av_attr avAttr = {};
    avAttr.type = FI_AV_TABLE;
    int rc = fi_av_open(ctx.domain(), &avAttr, &av_, nullptr);
    checkOfi(rc, "fi_av_open");
    DEBUG(NET, "OfiEndpointResources: AV created for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name);

    fi_cq_attr cqAttr = {};
    cqAttr.format = FI_CQ_FORMAT_CONTEXT;
    //cqAttr.format = FI_CQ_FORMAT_MSG;
    cqAttr.wait_obj = FI_WAIT_NONE;
    //cqAttr.size = (config.maxWriteQueueSize > 0) ? static_cast<size_t>(config.maxWriteQueueSize) : 1024;
    cqAttr.size = std::max<size_t>(qsz, epInfo->tx_attr->size + epInfo->rx_attr->size);

    logOfiEndpointConfigOnce(ctx.info(), epInfo, qsz, cqAttr.size, kOfiBindFlagsCq, kOfiBindFlagsCntr);

    rc = fi_cq_open(ctx.domain(), &cqAttr, &cq_, nullptr);
    checkOfi(rc, "fi_cq_open");
    DEBUG(NET, "OfiEndpointResources: CQ created for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name);

    fi_cntr_attr cntrAttr = {};
    cntrAttr.events = FI_CNTR_EVENTS_COMP;
    cntrAttr.wait_obj = FI_WAIT_NONE;
    rc = fi_cntr_open(ctx.domain(), &cntrAttr, &txCntr_, nullptr);
    checkOfi(rc, "fi_cntr_open");
    DEBUG(NET, "OfiEndpointResources: CNTR created for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name);

    //rc = fi_endpoint(ctx.domain(), ctx.info(), &ep_, nullptr);
    rc = fi_endpoint(ctx.domain(), epInfo, &ep_, nullptr);
    checkOfi(rc, "fi_endpoint");
    DEBUG(NET, "OfiEndpointResources: EP created for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name);

    rc = fi_ep_bind(ep_, &av_->fid, 0);
    checkOfi(rc, "fi_ep_bind(AV)");
    DEBUG(NET, "OfiEndpointResources: EP bound to AV for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name);

    //rc = fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV);
    //rc = fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION);
    rc = fi_ep_bind(ep_, &cq_->fid, kOfiBindFlagsCq);
    //rc = fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT);
    checkOfi(rc, "fi_ep_bind(CQ)");
    DEBUG(NET, "OfiEndpointResources: EP bound to CQ for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name, " flags=", static_cast<unsigned long long>(kOfiBindFlagsCq));

    rc = fi_ep_bind(ep_, &txCntr_->fid, kOfiBindFlagsCntr);
    checkOfi(rc, "fi_ep_bind(CNTR)");
    DEBUG(NET, "OfiEndpointResources: EP bound to CNTR for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name, " flags=", static_cast<unsigned long long>(kOfiBindFlagsCntr));

    rc = fi_enable(ep_);
    checkOfi(rc, "fi_enable");
    INFO(NET, "OfiEndpointResources: EP enabled for provider ", ctx.info()->fabric_attr->prov_name,
         " domain=", ctx.info()->domain_attr->name);

    cacheAddress();
    DEBUG(NET, "OfiEndpointResources: address cached for provider ", ctx.info()->fabric_attr->prov_name,
          " domain=", ctx.info()->domain_attr->name, " addr_bytes=", addr_.size());

    supportsWriteData_ = false;
    if (supportsWriteData_) {
      flags_ |= kOfiEndpointFlagWriteData;
    }
    //{
    //  size_t injectRmaSize = 0;
    //  size_t optlen = sizeof(injectRmaSize);
    //  int rc = fi_getopt(&ep_->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE, &injectRmaSize, &optlen);
    //  if (rc == -FI_ENOPROTOOPT) {
    //    injectRmaSize = ctx.info()->tx_attr->inject_size;
    //  }
    //  std::cout << "OFI inject sizes: tx_attr->inject_size=" << ctx.info()->tx_attr->inject_size << " inject_rma_size=" << injectRmaSize << std::endl;
    //  INFO(CONN, "OFI inject sizes: tx_attr->inject_size=", ctx.info()->tx_attr->inject_size,
    //       " inject_rma_size=", injectRmaSize);
    //}

    fi_freeinfo(epInfo);
    epInfo = nullptr;
  } catch (...) {
    if (epInfo != nullptr) fi_freeinfo(epInfo);
    closeAll();
    throw;
  }
#endif  // defined(MSCCLPP_USE_OFI)
}

void OfiEndpointResources::cacheAddress() {
#if defined(MSCCLPP_USE_OFI)
  size_t addrLen = 0;
  int rc = fi_getname(&ep_->fid, nullptr, &addrLen);
  if (rc != -FI_ETOOSMALL) {
    checkOfi(rc, "fi_getname(size)");
  }

  addr_.resize(addrLen);
  rc = fi_getname(&ep_->fid, addr_.data(), &addrLen);
  checkOfi(rc, "fi_getname");
  addr_.resize(addrLen);
#endif  // defined(MSCCLPP_USE_OFI)
}

void OfiEndpointResources::closeAll() noexcept {
#if defined(MSCCLPP_USE_OFI)
  if (ep_ != nullptr) {
    fi_close(&ep_->fid);
    ep_ = nullptr;
  }
  if (txCntr_ != nullptr) {
    fi_close(&txCntr_->fid);
    txCntr_ = nullptr;
  }
  if (cq_ != nullptr) {
    fi_close(&cq_->fid);
    cq_ = nullptr;
  }
  if (av_ != nullptr) {
    fi_close(&av_->fid);
    av_ = nullptr;
  }
  addr_.clear();
  flags_ = 0;
  supportsWriteData_ = false;
#endif  // defined(MSCCLPP_USE_OFI)
}

}  // namespace mscclpp
