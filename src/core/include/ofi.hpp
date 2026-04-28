#ifndef MSCCLPP_OFI_HPP_
#define MSCCLPP_OFI_HPP_

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <vector>

// Forward declarations for libfabric types.
struct fi_info;
struct fid_fabric;
struct fid_domain;
struct fid_ep;
struct fid_av;
struct fid_cq;
struct fid_mr;

namespace mscclpp {

struct OfiEndpointWireInfo {
  uint32_t version = 1;
  uint32_t flags = 0;
  std::vector<uint8_t> addr;
};

struct OfiMrInfo {
  uint64_t addr = 0;
  uint64_t rkey = 0;
  uint64_t size = 0;
};

enum class OfiHmemIface {
  System,
  Cuda,
  Rocr
};

struct OfiMemoryAttr {
  OfiHmemIface iface = OfiHmemIface::System;
  int device = -1;
  bool deviceOnly = false;
};

OfiMemoryAttr classifyOfiMemory(void* data);

class OfiCtx;

class OfiEndpointResources {
 public:
  OfiEndpointResources(OfiCtx& ctx, EndpointConfig const& config);
  ~OfiEndpointResources() { closeAll(); }

  OfiEndpointResources(OfiEndpointResources const&) = delete;
  OfiEndpointResources& operator=(OfiEndpointResources const&) = delete;

  OfiEndpointResources(OfiEndpointResources&&) = delete;
  OfiEndpointResources& operator=(OfiEndpointResources&&) = delete;

  OfiCtx& ctx() const { return *ctx_; }
  fid_ep* ep() const { return ep_; }
  fid_av* av() const { return av_; }
  fid_cq* cq() const { return cq_; }

  std::vector<uint8_t> const& address() const { return addr_; }
  uint32_t flags() const { return flags_; }
  bool supportsWriteData() const { return supportsWriteData_; }

 private:
  void closeAll() noexcept;
  void cacheAddress();

  OfiCtx* ctx_;
  fid_ep* ep_ = nullptr;
  fid_av* av_ = nullptr;
  fid_cq* cq_ = nullptr;

  std::vector<uint8_t> addr_;
  uint32_t flags_ = 0;
  bool supportsWriteData_ = false;
};


//class OfiMr {
// public:
//  OfiMr(fid_domain* domain, void* data, size_t size);
//  ~OfiMr() { close(); }
//
//  OfiMr(OfiMr const&) = delete;
//  OfiMr& operator=(OfiMr const&) = delete;
//  OfiMr(OfiMr&&) = delete;
//  OfiMr& operator=(OfiMr&&) = delete;
//
//  fid_mr* mr() const { return mr_; }
//  void* desc() const;
//  OfiMrInfo const& getInfo() const { return info_; }
//
// private:
//  void close() noexcept;
//
//  fid_mr* mr_ = nullptr;
//  OfiMrInfo info_{};
//};
class OfiMr {
 public:
  OfiMr(OfiEndpointResources& epRes, void* data, size_t size, OfiMemoryAttr const& memAttr);
  ~OfiMr() { close(); }

  OfiMr(OfiMr const&) = delete;
  OfiMr& operator=(OfiMr const&) = delete;
  OfiMr(OfiMr&&) = delete;
  OfiMr& operator=(OfiMr&&) = delete;

  fid_mr* mr() const { return mr_; }
  void* desc() const;
  OfiMrInfo const& getInfo() const { return info_; }

 private:
  void close() noexcept;

  fid_mr* mr_ = nullptr;
  OfiMrInfo info_{};
};

class OfiCtx {
 public:
  explicit OfiCtx(EndpointConfig::Ofi const& config);
  ~OfiCtx() { closeAll(); }

  OfiCtx(OfiCtx const&) = delete;
  OfiCtx& operator=(OfiCtx const&) = delete;
  OfiCtx(OfiCtx&&) = delete;
  OfiCtx& operator=(OfiCtx&&) = delete;

  fi_info* info() const { return info_; }
  fid_fabric* fabric() const { return fabric_; }
  fid_domain* domain() const { return domain_; }
  uint64_t mrMode() const { return mrMode_; }
  uint64_t caps() const { return caps_; }

  //std::unique_ptr<const OfiMr> registerMr(void* data, size_t size) const;

 private:
  void closeAll() noexcept;

  fi_info* info_ = nullptr;
  fid_fabric* fabric_ = nullptr;
  fid_domain* domain_ = nullptr;
  uint64_t mrMode_ = 0;
  uint64_t caps_ = 0;
};

}  // namespace mscclpp

#endif  // MSCCLPP_OFI_HPP_
