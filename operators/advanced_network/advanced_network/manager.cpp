/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "advanced_network/manager.h"
// Include the appropriate headers based on which ANO_MGR types are defined
#if ANO_MGR_DPDK
#include "advanced_network/managers/dpdk/adv_network_dpdk_mgr.h"
#endif
#if ANO_MGR_GPUNETIO
#include "advanced_network/managers/gpunetio/adv_network_doca_mgr.h"
#endif
#if ANO_MGR_RIVERMAX
#include "advanced_network/managers/rivermax/adv_network_rivermax_mgr.h"
#endif

#if ANO_MGR_DPDK || ANO_MGR_GPUNETIO
#include <rte_common.h>
#include <rte_malloc.h>
#endif

#include "holoscan/holoscan.hpp"

namespace holoscan::advanced_network {

// Initialize static members
std::unique_ptr<Manager> ManagerFactory::ManagerInstance_ = nullptr;  // Initialize static members
ManagerType ManagerFactory::ManagerType_ = ManagerType::UNKNOWN;

extern void initialize_manager(Manager* _manager);

ManagerType ManagerFactory::get_default_manager_type() {
  ManagerType mgr_type = ManagerType::UNKNOWN;
#if ANO_MGR_DPDK
  mgr_type = ManagerType::DPDK;
#elif ANO_MGR_GPUNETIO
  mgr_type = ManagerType::DOCA;
#elif ANO_MGR_RIVERMAX
  mgr_type = ManagerType::RIVERMAX;
#else
#error "No Advanced Network manager defined"
#endif
  return mgr_type;
}

std::unique_ptr<Manager> ManagerFactory::create_instance(ManagerType type) {
  std::unique_ptr<Manager> _manager;
  switch (type) {
#if ANO_MGR_DPDK
    case ManagerType::DPDK:
      _manager = std::make_unique<DpdkMgr>();
      break;
#endif
#if ANO_MGR_GPUNETIO
    case ManagerType::DOCA:
      _manager = std::make_unique<DocaMgr>();
      break;
#endif
#if ANO_MGR_RIVERMAX
    case ManagerType::RIVERMAX:
      _manager = std::make_unique<RivermaxMgr>();
      break;
#endif
    case ManagerType::DEFAULT:
      _manager = create_instance(get_default_manager_type());
      return _manager;
    case ManagerType::UNKNOWN:
      throw std::invalid_argument("Unknown manager type");
    default:
      throw std::invalid_argument("Invalid type");
  }

  // Initialize the ADV Net Common API
  initialize_manager(_manager.get());
  return _manager;
}

template <typename Config>
ManagerType ManagerFactory::get_manager_type(const Config& config) {
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
      return manager_type_from_string(holoscan::advanced_network::ANO_MGR_STR__DEFAULT);
    }
  }

  return manager_type_from_string(holoscan::advanced_network::ANO_MGR_STR__DEFAULT);
}

template ManagerType ManagerFactory::get_manager_type<Config>(const Config&);

Status Manager::allocate_memory_regions() {
  HOLOSCAN_LOG_INFO("Registering memory regions");
#if ANO_MGR_DPDK || ANO_MGR_GPUNETIO
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
            return Status::NULL_PTR;
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
            const char* err_str = nullptr;
            cuGetErrorString(alloc_res, &err_str);
            HOLOSCAN_LOG_CRITICAL("Could not allocate {:.2f}MB of GPU memory. Error: {}",
                                  align / 1e6,
                                  err_str);
            return Status::NULL_PTR;
          }

          ptr = reinterpret_cast<void*>(cuptr);

          const auto attr_res =
              cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, cuptr);
          if (attr_res != CUDA_SUCCESS) {
            HOLOSCAN_LOG_CRITICAL("Could not set pointer attributes");
            return Status::NULL_PTR;
          }
          break;
        }
        default:
          HOLOSCAN_LOG_ERROR("Unknown memory type {}!", static_cast<int>(mr.second.kind_));
          return Status::INVALID_PARAMETER;
      }

      if (ptr == nullptr) {
        HOLOSCAN_LOG_CRITICAL("Fatal to allocate {} of type {} for MR",
                              mr.second.ttl_size_,
                              static_cast<int>(mr.second.kind_));
        return Status::NULL_PTR;
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
  return Status::SUCCESS;
}

/**
 * @brief Generic implementation of get_port_id that looks up port in config
 * This is a final method that cannot be overridden by subclasses.
 *
 * @param key PCIe address or config name of the interface to look up
 * @return int Port ID or -1 if not found
 */
int Manager::get_port_id(const std::string& key) {
  for (const auto& intf : cfg_.ifs_) {
    if (intf.address_ == key) { return intf.port_id_; }
    if (intf.name_ == key) { return intf.port_id_; }
  }
  return -1;
}

bool Manager::validate_config() const {
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

void Manager::init_rx_core_q_map() {
  for (const auto& intf : cfg_.ifs_) {
    // Initialize the round-robin index for this port
    next_queue_index_map_.try_emplace(intf.port_id_, 0);

    for (const auto& q : intf.rx_.queues_) {
      int cpu_core = strtol(q.common_.cpu_core_.c_str(), nullptr, 10);
      rx_core_q_map[cpu_core].push_back(std::make_pair(intf.port_id_, q.common_.id_));

      if (rx_core_q_map[cpu_core].size() > MAX_RX_Q_PER_CORE) {
        HOLOSCAN_LOG_CRITICAL("Too many RX queues assigned to core {}!", cpu_core);
      }
    }
  }
}

uint16_t Manager::get_num_rx_queues(int port_id) const {
  return cfg_.ifs_[port_id].rx_.queues_.size();
}

void Manager::flush_port_queue(int port, int queue) {
  HOLOSCAN_LOG_ERROR("flush_port_queue not implemented for this manager type");
}

void* Manager::drop_all_traffic(int port) {
  HOLOSCAN_LOG_ERROR("drop_all_traffic not implemented for this manager type");
  return nullptr;
}

Status Manager::allow_all_traffic(int port, void* flow) {
  HOLOSCAN_LOG_ERROR("allow_all_traffic not implemented for this manager type");
  return Status::NOT_SUPPORTED;
}

Status Manager::get_rx_burst(BurstParams** burst, int port_id) {
  // Check if the port_id is valid
  if (port_id < 0 || port_id >= static_cast<int>(cfg_.ifs_.size())) {
    HOLOSCAN_LOG_ERROR("Invalid port_id {} provided to get_rx_burst", port_id);
    return Status::INVALID_PARAMETER;
  }

  const auto& queues = cfg_.ifs_[port_id].rx_.queues_;
  size_t num_queues = queues.size();
  size_t& next_queue_index = next_queue_index_map_[port_id];

  // Check all queues once, starting from the next index
  for (size_t i = 0; i < num_queues; ++i) {
    size_t check_index = (next_queue_index + i) % num_queues;
    int queue_id = queues[check_index].common_.id_;

    Status ret = get_rx_burst(burst, port_id, queue_id);
    if (ret != Status::NULL_PTR) {
      // Got something, update index for next time and return status
      next_queue_index = (check_index + 1) % num_queues;
      return ret;
    }
  }

  // If we checked all queues and none had data
  return Status::NULL_PTR;
}

Status Manager::get_rx_burst(BurstParams** burst) {
  if (cfg_.ifs_.empty()) {
    HOLOSCAN_LOG_ERROR("No interfaces configured");
    return Status::NULL_PTR;
  }

  size_t num_interfaces = cfg_.ifs_.size();

  // Check all queues once, starting from the next index
  for (size_t i = 0; i < num_interfaces; ++i) {
    size_t check_index = (next_port_index_ + i) % num_interfaces;
    int port_id = cfg_.ifs_[check_index].port_id_;

    Status ret = get_rx_burst(burst, port_id);
    if (ret != Status::NULL_PTR) {
      // Got something, update index for next time and return status
      next_port_index_ = (check_index + 1) % num_interfaces;
      return ret;
    }
  }

  // If we checked all interfaces and none yielded a burst
  return Status::NULL_PTR;
}

};  // namespace holoscan::advanced_network
