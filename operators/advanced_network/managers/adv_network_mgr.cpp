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
#include "adv_network_dpdk_mgr.h"
#include "adv_network_doca_mgr.h"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

ANOMgr* g_ano_mgr = nullptr;

/* This function decides what ANO backend or "manager" is used for packet processing. The choice of
   manager is based on what we believe is the best selection based on the user's configuration. */
void set_ano_mgr(const AdvNetConfigYaml& cfg) {
  if (g_ano_mgr == nullptr) {
    if (cfg.common_.mgr_ == "doca") {
      HOLOSCAN_LOG_INFO("Selecting DOCA as ANO manager");
      g_ano_mgr = new DocaMgr{};
    } else {
      HOLOSCAN_LOG_INFO("Selecting DPDK as ANO manager");
      g_ano_mgr = new DpdkMgr{};
    }
  }
}

AdvNetStatus ANOMgr::allocate_memory_regions() {
  HOLOSCAN_LOG_INFO("Registering memory regions");

  for (const auto& mr : cfg_.mrs_) {
    void* ptr;
    AllocRegion ar;
    size_t buf_size = mr.second.buf_size_ * mr.second.num_bufs_;

    if (buf_size & 0x3) {
      HOLOSCAN_LOG_CRITICAL("Total buffer size must be multiple of 4 for MR {}", mr.second.name_);
      return AdvNetStatus::NULL_PTR;
    }
    if (mr.second.owned_) {
      switch (mr.second.kind_) {
        case MemoryKind::HOST:
          ptr = malloc(buf_size);
          break;
        case MemoryKind::HOST_PINNED:
          if (cudaHostAlloc(&ptr, buf_size, 0) != cudaSuccess) {
            HOLOSCAN_LOG_CRITICAL("Failed to allocate CUDA pinned host memory!");
            return AdvNetStatus::NULL_PTR;
          }
          break;
        case MemoryKind::HUGE:
          ptr = rte_malloc_socket(nullptr, buf_size, RTE_PKTMBUF_HEADROOM, mr.second.affinity_);
          break;
        case MemoryKind::DEVICE: {
          unsigned int flag = 1;
          const auto align = RTE_ALIGN_CEIL(buf_size, GPU_PAGE_SIZE);
          CUdeviceptr cuptr;

          cudaSetDevice(mr.second.affinity_);
          cudaFree(0);  // Create primary context if it doesn't exist
          const auto alloc_res = cuMemAlloc(&cuptr, align);
          if (alloc_res != CUDA_SUCCESS) {
            HOLOSCAN_LOG_CRITICAL(
                "Could not allocate {:.2f}MB of GPU memory: {}", align / 1e6, alloc_res);
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
        HOLOSCAN_LOG_CRITICAL(
            "Fatal to allocate {} of type {} for MR", buf_size, static_cast<int>(mr.second.kind_));
        return AdvNetStatus::NULL_PTR;
      }
    }

    HOLOSCAN_LOG_INFO(
        "Successfully allocated memory region {} at {} with {} bytes ({} elements @ {} bytes)",
        mr.second.name_,
        ptr,
        buf_size,
        mr.second.num_bufs_,
        mr.second.buf_size_);
    ar_[mr.second.name_] = {mr.second.name_, ptr};
  }

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
