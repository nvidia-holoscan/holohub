/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda.h>
#include "adv_network_mgr.h"
// Include the appropriate headers based on which ANO_MGR types are defined
#if ANO_MGR_DPDK
#include "adv_network_dpdk_mgr.h"
#endif
#if ANO_MGR_DOCA
#include "adv_network_doca_mgr.h"
#endif
#if ANO_MGR_RIVERMAX
#include "adv_network_rmax_mgr.h"
#endif

#if ANO_MGR_DPDK || ANO_MGR_DOCA
#include <rte_common.h>
#include <rte_malloc.h>
#endif

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

// Initialize static members
std::unique_ptr<ANOMgr> AnoMgrFactory::AnoMgrInstance_ = nullptr;  // Initialize static members
AnoMgrType AnoMgrFactory::AnoMgrType_ = AnoMgrType::UNKNOWN;

extern void adv_net_initialize_manager(ANOMgr* _manager);

AnoMgrType AnoMgrFactory::get_default_manager_type() {
  AnoMgrType mgr_type = AnoMgrType::UNKNOWN;
#if ANO_MGR_DPDK
  mgr_type = AnoMgrType::DPDK;
#elif ANO_MGR_DOCA
  mgr_type = AnoMgrType::DOCA;
#elif ANO_MGR_RIVERMAX
  mgr_type = AnoMgrType::RIVERMAX;
#else
#error "No advanced network operator manager defined"
#endif
  return mgr_type;
}

std::unique_ptr<ANOMgr> AnoMgrFactory::create_instance(AnoMgrType type) {
  std::unique_ptr<ANOMgr> _manager;
  switch (type) {
#if ANO_MGR_DPDK
    case AnoMgrType::DPDK:
      _manager = std::make_unique<DpdkMgr>();
      break;
#endif
#if ANO_MGR_DOCA
    case AnoMgrType::DOCA:
      _manager = std::make_unique<DocaMgr>();
      break;
#endif
#if ANO_MGR_RIVERMAX
    case AnoMgrType::RIVERMAX:
      _manager = std::make_unique<RmaxMgr>();
      break;
#endif
    case AnoMgrType::DEFAULT:
      _manager = create_instance(get_default_manager_type());
      return _manager;
    case AnoMgrType::UNKNOWN:
      throw std::invalid_argument("Unknown manager type");
    default:
      throw std::invalid_argument("Invalid type");
  }

  // Initialize the ADV Net Common API
  adv_net_initialize_manager(_manager.get());
  return _manager;
}

template <typename Config>
AnoMgrType AnoMgrFactory::get_manager_type(const Config& config) {
  // Ensure that Config has a method yaml_nodes() that returns a collection
  // of YAML nodes
  static_assert(
      std::is_member_function_pointer<decltype(&Config::yaml_nodes)>::value,
      "Config type must have a method yaml_nodes() that returns a collection of YAML nodes");

  auto& yaml_nodes = config.yaml_nodes();
  for (const auto& yaml_node : yaml_nodes) {
    try {
      auto node = yaml_node["advanced_network"]["cfg"];
      std::string manager = node["manager"].template as<std::string>(ANO_MGR_STR__DEFAULT);
      return manager_type_from_string(manager);
    } catch (const std::exception& e) {
      return manager_type_from_string(holoscan::ops::ANO_MGR_STR__DEFAULT);
    }
  }

  return manager_type_from_string(holoscan::ops::ANO_MGR_STR__DEFAULT);
}

template AnoMgrType AnoMgrFactory::get_manager_type<Config>(const Config&);

AdvNetStatus ANOMgr::allocate_memory_regions() {
  HOLOSCAN_LOG_INFO("Registering memory regions");
#if ANO_MGR_DPDK || ANO_MGR_DOCA
  for (auto& mr : cfg_.mrs_) {
    void* ptr;
    AllocRegion ar;
    mr.second.ttl_size_ = RTE_ALIGN_CEIL(mr.second.adj_size_ * mr.second.num_bufs_, GPU_PAGE_SIZE);

    if (mr.second.owned_) {
      switch (mr.second.kind_) {
        case MemoryKind::HOST:
          ptr = malloc(mr.second.ttl_size_);
          break;
        case MemoryKind::HOST_PINNED:
          if (cudaHostAlloc(&ptr, mr.second.ttl_size_, 0) != cudaSuccess) {
            HOLOSCAN_LOG_CRITICAL("Failed to allocate CUDA pinned host memory!");
            return AdvNetStatus::NULL_PTR;
          }
          break;
        case MemoryKind::HUGE:
          ptr = rte_malloc_socket(nullptr, mr.second.ttl_size_, 0, mr.second.affinity_);
          break;
        case MemoryKind::DEVICE: {
          unsigned int flag = 1;
          const auto align = RTE_ALIGN_CEIL(mr.second.ttl_size_, GPU_PAGE_SIZE);
          CUdeviceptr cuptr;

          cudaSetDevice(mr.second.affinity_);
          cudaFree(0);  // Create primary context if it doesn't exist
          const auto alloc_res = cuMemAlloc(&cuptr, align);
          if (alloc_res != CUDA_SUCCESS) {
            HOLOSCAN_LOG_CRITICAL("Could not allocate {:.2f}MB of GPU memory: {}",
                                  align / 1e6,
                                  static_cast<int>(alloc_res));
            return AdvNetStatus::NULL_PTR;
          }

          ptr = reinterpret_cast<void*>(cuptr);

          const auto attr_res =
              cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, cuptr);
          if (attr_res != CUDA_SUCCESS) {
            HOLOSCAN_LOG_CRITICAL("Could not set pointer attributes");
            return AdvNetStatus::NULL_PTR;
          }
          break;
        }
        default:
          HOLOSCAN_LOG_ERROR("Unknown memory type {}!", static_cast<int>(mr.second.kind_));
          return AdvNetStatus::INVALID_PARAMETER;
      }

      if (ptr == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Fatal to allocate {} of type {} for MR",
                              mr.second.ttl_size_,
                              static_cast<int>(mr.second.kind_));
        return AdvNetStatus::NULL_PTR;
      }
    }

    HOLOSCAN_LOG_INFO(
        "Successfully allocated memory region {} at {} type {} with {} bytes "
        "({} elements @ {} bytes total {})",
        mr.second.name_,
        ptr,
        (int)mr.second.kind_,
        mr.second.buf_size_,
        mr.second.num_bufs_,
        mr.second.adj_size_,
        mr.second.ttl_size_);
    ar_[mr.second.name_] = {mr.second.name_, ptr};
  }
#endif
  HOLOSCAN_LOG_INFO("Finished allocating memory regions");
  return AdvNetStatus::SUCCESS;
}

bool ANOMgr::validate_config() const {
  bool pass = true;
  std::set<std::string> mr_names;
  std::set<std::string> q_mr_names;

  // Verify all memory regions are used in queues and all queue MRs are listed in the MR section
  for (const auto& mr : cfg_.mrs_) { mr_names.emplace(mr.second.name_); }

  for (const auto& intf : cfg_.ifs_) {
    for (const auto& rxq : intf.rx_.queues_) {
      for (const auto& mr : rxq.common_.mrs_) { q_mr_names.emplace(mr); }
    }
    for (const auto& txq : intf.tx_.queues_) {
      for (const auto& mr : txq.common_.mrs_) { q_mr_names.emplace(mr); }
    }
  }

  // All MRs are in queues
  for (const auto& mr : mr_names) {
    if (q_mr_names.find(mr) == q_mr_names.end()) {
      HOLOSCAN_LOG_WARN("Extra MR section with name {} unused in queues section", mr);
    }
  }

  // All queue MRs are in MR list
  for (const auto& mr : q_mr_names) {
    if (mr_names.find(mr) == mr_names.end()) {
      HOLOSCAN_LOG_ERROR(
          "Queue found using MR {}, but that MR doesn't exist in the memory_region config", mr);
      pass = false;
    }
  }

  return pass;
}

};  // namespace holoscan::ops
