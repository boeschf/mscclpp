#include <mscclpp/gpu_utils.hpp>

#include "ofi.hpp"

#if defined(MSCCLPP_USE_OFI)
#include "ofi_wrapper.hpp"
#endif  // defined(MSCCLPP_USE_OFI)

#include <cstring>
#include <mscclpp/errors.hpp>

#include "logger.hpp"
#include <iostream>

namespace mscclpp {


namespace {

#if defined(MSCCLPP_USE_OFI)

constexpr std::uint32_t kOfiEndpointFlagWriteData = 1u << 0;

[[noreturn]] void throwOfiError(char const* what, int rc) {
  THROW(NET, Error, ErrorCode::SystemError, what, " failed: ", fi_strerror(-rc), " (rc=", rc, ")");
}

void checkOfi(int rc, char const* what) {
  if (rc != 0) {
    throwOfiError(what, rc);
  }
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
  std::cout << "        OfiMr: registering memory for address " << data << " with size " << size << " using  fi_mr_reg" << std::endl;
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
    attr.access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
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
                   FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
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

  info_.addr = reinterpret_cast<uint64_t>(data);
  info_.rkey = fi_mr_key(mr_);
  info_.size = static_cast<uint64_t>(size);
  std::cout << "        OfiMr: registered memory with addr " << info_.addr << " and rkey " << info_.rkey << std::endl;
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
  std::cout << "      OfiCtx: allocated hints for provider " << config.provider << " and domain " << config.domain << std::endl;

  try {
    hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

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

    std::cout << "      OfiCtx: attempting fi_getinfo for provider " << config.provider << " and domain " << config.domain << std::endl;
    // Keep this at a generic version for endpoint/domain bring-up.
    int rc = fi_getinfo(FI_VERSION(2, 5), nullptr, nullptr, 0, hints, &info_);
    fi_freeinfo(hints);
    hints = nullptr;
    std::cout << "      checking ofi" << std::endl;
    checkOfi(rc, "fi_getinfo(OfiCtx)");
    std::cout << "      OfiCtx: fi_getinfo succeeded for provider " << config.provider << " and domain " << config.domain << std::endl;

    caps_ = info_->caps;
    mrMode_ = info_->domain_attr->mr_mode;

    rc = fi_fabric(info_->fabric_attr, &fabric_, nullptr);
    checkOfi(rc, "fi_fabric(OfiCtx)");
    std::cout << "      OfiCtx: fi_fabric succeeded for provider " << config.provider << " and domain " << config.domain << std::endl;

    rc = fi_domain(fabric_, info_, &domain_, nullptr);
    checkOfi(rc, "fi_domain(OfiCtx)");
    std::cout << "      OfiCtx: fi_domain succeeded for provider " << config.provider << " and domain " << config.domain << std::endl;
  } catch (...) {
    if (hints != nullptr) {
      fi_freeinfo(hints);
    }
    std::cout << "    OfiCtx: exception during initialization, closing any opened resources for provider " << config.provider
              << " and domain " << config.domain << std::endl;
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

  std::cout << "      OfiEndpointResources: creating AV for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;
  fi_av_attr avAttr = {};
  avAttr.type = FI_AV_TABLE;
  int rc = fi_av_open(ctx.domain(), &avAttr, &av_, nullptr);
  checkOfi(rc, "fi_av_open");
  std::cout << "      OfiEndpointResources: AV created for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;

  fi_cq_attr cqAttr = {};
  cqAttr.format = FI_CQ_FORMAT_CONTEXT;
  cqAttr.wait_obj = FI_WAIT_NONE;
  cqAttr.size = (config.maxWriteQueueSize > 0) ? static_cast<size_t>(config.maxWriteQueueSize) : 1024;
  rc = fi_cq_open(ctx.domain(), &cqAttr, &cq_, nullptr);
  checkOfi(rc, "fi_cq_open");
  std::cout << "      OfiEndpointResources: CQ created for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;

  rc = fi_endpoint(ctx.domain(), ctx.info(), &ep_, nullptr);
  checkOfi(rc, "fi_endpoint");
  std::cout << "      OfiEndpointResources: EP created for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;

  rc = fi_ep_bind(ep_, &av_->fid, 0);
  checkOfi(rc, "fi_ep_bind(AV)");
  std::cout << "      OfiEndpointResources: EP bound to AV for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;

  rc = fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV);
  checkOfi(rc, "fi_ep_bind(CQ)");
  std::cout << "      OfiEndpointResources: EP bound to CQ for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;

  rc = fi_enable(ep_);
  checkOfi(rc, "fi_enable");
  std::cout << "      OfiEndpointResources: EP enabled for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;

  cacheAddress();
  std::cout << "      OfiEndpointResources: address cached for provider " << ctx.info()->fabric_attr->prov_name
            << " and domain " << ctx.info()->domain_attr->name << std::endl;

  supportsWriteData_ = false;
  if (supportsWriteData_) {
    flags_ |= kOfiEndpointFlagWriteData;
  }
  {
    size_t injectRmaSize = 0;
    size_t optlen = sizeof(injectRmaSize);
    int rc = fi_getopt(&ep_->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE, &injectRmaSize, &optlen);
    if (rc == -FI_ENOPROTOOPT) {
      injectRmaSize = ctx.info()->tx_attr->inject_size;
    }
    std::cout << "OFI inject sizes: tx_attr->inject_size=" << ctx.info()->tx_attr->inject_size << " inject_rma_size=" << injectRmaSize << std::endl;
    INFO(CONN, "OFI inject sizes: tx_attr->inject_size=", ctx.info()->tx_attr->inject_size,
         " inject_rma_size=", injectRmaSize);
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
