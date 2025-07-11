/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "adv_network_dpdk_stats.h"
#include "holoscan/holoscan.hpp"
#include <thread>
#include <chrono>

namespace holoscan::advanced_network {

void DpdkStats::Init(const NetworkConfig &cfg) {
  cfg_ = cfg;
  init_ = true;

  HOLOSCAN_LOG_INFO("Initializing DPDK stats");

  // Populate the memory region names map for each port/queue combination
  port_queue_memory_regions_.clear();
  for (const auto &intf : cfg_.ifs_) {
    int port_id = intf.port_id_;

    // Process RX queues
    for (const auto &q : intf.rx_.queues_) {
      uint32_t key = (port_id << 16) | q.common_.id_;
      std::string mr_names;

      // Concatenate memory region names
      for (size_t i = 0; i < q.common_.mrs_.size(); ++i) {
        if (i > 0) {
          mr_names += ", ";
        }
        mr_names += q.common_.mrs_[i];
      }

      port_queue_memory_regions_[key] = mr_names;
      HOLOSCAN_LOG_INFO("Port {}, Queue {}: Memory regions: {}",
                        port_id, q.common_.id_, mr_names);
    }
  }

  // Initialize xstats for each port
  for (const auto &port : cfg_.ifs_) {
    int port_id = port.port_id_;
    PortXStats& port_stats = xstats_[port_id];

    // Get the number of xstats
    int len = rte_eth_xstats_get(port_id, NULL, 0);
    if (len < 0) {
      HOLOSCAN_LOG_ERROR("rte_eth_xstats_get({}) failed: {}", port_id, len);
      continue;
    }

    // Allocate memory for xstats
    port_stats.len = len;
    port_stats.xstats = (struct rte_eth_xstat*)calloc(len, sizeof(struct rte_eth_xstat));
    port_stats.old_xstats = (struct rte_eth_xstat*)calloc(len, sizeof(struct rte_eth_xstat));

    if (port_stats.xstats == NULL || port_stats.old_xstats == NULL) {
      HOLOSCAN_LOG_ERROR("Failed to allocate memory for xstats for port {}", port_id);
      if (port_stats.xstats) free(port_stats.xstats);
      if (port_stats.old_xstats) free(port_stats.old_xstats);
      continue;
    }

    // Get xstats names
    struct rte_eth_xstat_name *xstats_names =
        (struct rte_eth_xstat_name*)calloc(len, sizeof(struct rte_eth_xstat_name));
    if (xstats_names == NULL) {
      HOLOSCAN_LOG_ERROR("Failed to allocate memory for xstats_names for port {}", port_id);
      free(port_stats.xstats);
      free(port_stats.old_xstats);
      continue;
    }

    int ret = rte_eth_xstats_get_names(port_id, xstats_names, len);
    if (ret < 0 || ret > len) {
      HOLOSCAN_LOG_ERROR("rte_eth_xstats_get_names({}) failed: {}", port_id, ret);
      free(xstats_names);
      free(port_stats.xstats);
      free(port_stats.old_xstats);
      continue;
    }

    // Create a map from stat name to index and find indices for rx_missed and rx_mbuf_alloc_err
    port_stats.rx_missed_idx = -1;
    port_stats.rx_mbuf_allocation_errors_idx = -1;
    port_stats.rx_queue_errors_idx.clear();

    for (int i = 0; i < len; i++) {
      port_stats.name_to_idx[xstats_names[i].name] = i;

      // Find indices for the stats we're interested in
      if (strcmp(xstats_names[i].name, "rx_missed_errors") == 0) {
        port_stats.rx_missed_idx = i;
      } else if (strcmp(xstats_names[i].name, "rx_mbuf_allocation_errors") == 0) {
        port_stats.rx_mbuf_allocation_errors_idx = i;
      }

      // Check for rx_q*_errors counters (queue-specific error counters)
      const char* rx_q_errors_prefix = "rx_q";
      const char* rx_q_errors_suffix = "_errors";
      if (strncmp(xstats_names[i].name, rx_q_errors_prefix, strlen(rx_q_errors_prefix)) == 0) {
        const char* queue_num_str = xstats_names[i].name + strlen(rx_q_errors_prefix);
        char* endptr;
        int queue_id = strtol(queue_num_str, &endptr, 10);

        // Verify this is a valid rx_q*_errors counter
        if (endptr != queue_num_str && strcmp(endptr, rx_q_errors_suffix) == 0) {
          if (queue_id >= 0 && queue_id < port_stats.MAX_QUEUE_COUNT) {
            port_stats.rx_queue_errors_idx[queue_id] = i;
            HOLOSCAN_LOG_INFO("Found rx_q{}_errors counter at index {}", queue_id, i);
          }
        }
      }
    }

    // Get initial xstats values
    ret = rte_eth_xstats_get(port_id, port_stats.xstats, len);
    if (ret < 0 || ret > len) {
      HOLOSCAN_LOG_ERROR("rte_eth_xstats_get({}) failed: {}", port_id, ret);
      free(xstats_names);
      free(port_stats.xstats);
      free(port_stats.old_xstats);
      continue;
    }

    // Copy initial values to old_xstats
    memcpy(port_stats.old_xstats, port_stats.xstats, len * sizeof(struct rte_eth_xstat));

    // Free xstats_names as we don't need them anymore
    free(xstats_names);

    HOLOSCAN_LOG_INFO("Initialized DPDK xstats for port {}, found {} stats", port_id, len);
    if (port_stats.rx_missed_idx >= 0) {
      HOLOSCAN_LOG_INFO("Found rx_missed counter at index {}", port_stats.rx_missed_idx);
    } else {
      HOLOSCAN_LOG_WARN("Could not find rx_missed counter for port {}", port_id);
    }

    if (port_stats.rx_mbuf_allocation_errors_idx >= 0) {
      HOLOSCAN_LOG_INFO("Found mbuf allocation counter at index {}",
        port_stats.rx_mbuf_allocation_errors_idx);
    } else {
      HOLOSCAN_LOG_WARN("Could not find mbuf allocation counter for port {}", port_id);
    }
  }

  init_ = true;

  HOLOSCAN_LOG_INFO("Initialized DPDK stats");
}

void DpdkStats::Shutdown() {
  init_ = false;

  force_quit_.store(true);
}

/**
 * @brief Constantly polls and updates the stats for the DPDK manager. The result of this polling
   can be used by the user's application to check for errors or other issues.
 */
void DpdkStats::Run() {
  cpu_set_t cpuset;

  auto thread = pthread_self();
  CPU_ZERO(&cpuset);
  CPU_SET(cfg_.common_.master_core_, &cpuset);  // Set affinity to CPU core_

  int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    HOLOSCAN_LOG_ERROR("Stats thread pthread_setaffinity_np({}) failed: {}", core_, s);
    return;
  }

  pthread_setname_np(thread, "DPDK_STATS");
  HOLOSCAN_LOG_INFO("Starting stats thread on core {}", cfg_.common_.master_core_);

  while (!force_quit_.load()) {
    // Poll xstats for each port
    for (const auto &port : cfg_.ifs_) {
      int port_id = port.port_id_;
      const auto interface_name = port.name_;

      // Skip ports that weren't initialized
      auto port_it = xstats_.find(port_id);
      if (port_it == xstats_.end()) {
        continue;
      }

      PortXStats& port_stats = port_it->second;

      // Skip ports that weren't initialized
      if (port_stats.xstats == nullptr) {
        continue;
      }

      // Get current xstats values
      int ret = rte_eth_xstats_get(port_id, port_stats.xstats, port_stats.len);
      if (ret < 0 || ret > port_stats.len) {
        HOLOSCAN_LOG_ERROR("rte_eth_xstats_get({}) failed: {}", port_id, ret);
        continue;
      }

      // Check if rx_mbuf_alloc_err counter has increased
      if (port_stats.rx_mbuf_allocation_errors_idx >= 0) {
        uint64_t rx_mbuf_alloc_err =
          port_stats.xstats[port_stats.rx_mbuf_allocation_errors_idx].value;
        uint64_t old_rx_mbuf_alloc_err =
          port_stats.old_xstats[port_stats.rx_mbuf_allocation_errors_idx].value;

        if (rx_mbuf_alloc_err > old_rx_mbuf_alloc_err) {
          const auto alloc_err_since_last_check = rx_mbuf_alloc_err - old_rx_mbuf_alloc_err;
          HOLOSCAN_LOG_DEBUG("Port {}: Buffer allocation errors since last poll "\
          "{}ms ago: {}, total {}. Software is not keeping up with the NIC RX rate.",
          interface_name, POLLING_INTERVAL_MS, alloc_err_since_last_check, rx_mbuf_alloc_err);
        }
      }

      // Check if any rx_q*_errors counters have increased
      for (const auto& [queue_id, error_idx] : port_stats.rx_queue_errors_idx) {
        uint64_t rx_queue_errors = port_stats.xstats[error_idx].value;
        uint64_t old_rx_queue_errors = port_stats.old_xstats[error_idx].value;

        if (rx_queue_errors > old_rx_queue_errors) {
          // Get memory region names for this port/queue combination
          uint32_t key = (port_id << 16) | queue_id;
          std::string mr_names = "unknown";
          auto it = port_queue_memory_regions_.find(key);
          if (it != port_queue_memory_regions_.end()) {
            mr_names = it->second;
          }

          const auto q_cfg = port.rx_.queues_[queue_id].common_;

          HOLOSCAN_LOG_WARN("'{}' interface ({}), Rx '{}' queue ({}): "
          "Packets to '{}' might get dropped. Either CPU #{} couldn't read from the NIC fast "
          "enough, downstream software was not processing and freeing packets fast enough, "
          "or not enough buffers were allocated in this memory region to begin with.",
          interface_name, port_id, q_cfg.name_, queue_id, mr_names, q_cfg.cpu_core_);
        }
      }

      // Check if rx_missed counter has increased
      if (port_stats.rx_missed_idx >= 0) {
        uint64_t rx_missed = port_stats.xstats[port_stats.rx_missed_idx].value;
        uint64_t old_rx_missed = port_stats.old_xstats[port_stats.rx_missed_idx].value;

        if (rx_missed > old_rx_missed) {
          const auto missed_since_last_check = rx_missed - old_rx_missed;
          HOLOSCAN_LOG_ERROR(
            "'{}' interface ({}), Rx: Dropped {} packets since last poll {}ms ago (total: {})",
            interface_name, port_id, missed_since_last_check, POLLING_INTERVAL_MS, rx_missed);
        }
      }

      // Update old_xstats with current values
      memcpy(port_stats.old_xstats, port_stats.xstats,
          port_stats.len * sizeof(struct rte_eth_xstat));
    }

    // Sleep for a short time to avoid excessive CPU usage
    std::this_thread::sleep_for(std::chrono::milliseconds(POLLING_INTERVAL_MS));
  }
}

}  // namespace holoscan::advanced_network
