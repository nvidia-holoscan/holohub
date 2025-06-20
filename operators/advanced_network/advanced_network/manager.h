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

#pragma once

#include <rte_ring.h>
#include <rte_mbuf.h>
#include "advanced_network/types.h"
#include <optional>

namespace holoscan::advanced_network {

struct AllocRegion {
  std::string mr_name_;
  void* ptr_;
};

/**
 * @brief (Almost) ABC representing an interface into an advanced_network backend implementation
 *
 */
class Manager {
 public:
  static constexpr size_t MAX_RX_Q_PER_CORE = 16;

  virtual void initialize() = 0;
  virtual bool is_initialized() const { return initialized_; }
  virtual bool set_config_and_initialize(const NetworkConfig& cfg) = 0;
  virtual void run() = 0;

  // Common free functions to override
  virtual void* get_packet_ptr(BurstParams* burst, int idx) = 0;
  virtual uint32_t get_packet_length(BurstParams* burst, int idx) = 0;
  virtual void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx) = 0;
  virtual uint32_t get_segment_packet_length(BurstParams* burst, int seg, int idx) = 0;
  virtual uint16_t get_packet_flow_id(BurstParams* burst, int idx) = 0;
  virtual void* get_packet_extra_info(BurstParams* burst, int idx) = 0;
  virtual Status get_tx_packet_burst(BurstParams* burst) = 0;
  virtual Status set_eth_header(BurstParams* burst, int idx, char* dst_addr) = 0;
  virtual Status set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                                 unsigned int src_host, unsigned int dst_host) = 0;
  virtual Status set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                                uint16_t dst_port) = 0;
  virtual Status set_udp_payload(BurstParams* burst, int idx, void* data, int len) = 0;
  virtual bool is_tx_burst_available(BurstParams* burst) = 0;

  virtual Status set_packet_lengths(BurstParams* burst, int idx,
                                    const std::initializer_list<int>& lens) = 0;
  virtual void free_all_segment_packets(BurstParams* burst, int seg) = 0;
  virtual void free_all_packets(BurstParams* burst) = 0;
  virtual void free_packet_segment(BurstParams* burst, int seg, int pkt) = 0;
  virtual void free_packet(BurstParams* burst, int pkt) = 0;
  virtual void free_rx_burst(BurstParams* burst) = 0;
  virtual void free_tx_burst(BurstParams* burst) = 0;
  virtual Status set_packet_tx_time(BurstParams* burst, int idx, uint64_t time) = 0;
  virtual void shutdown() = 0;
  virtual void print_stats() = 0;
  virtual uint64_t get_burst_tot_byte(BurstParams* burst) = 0;
  virtual BurstParams* create_tx_burst_params() = 0;
  virtual Status get_rx_burst(BurstParams** burst, uintptr_t conn_id, bool server);
  virtual Status get_rx_burst(BurstParams** burst, int port, int q) = 0;
  virtual Status get_rx_burst(BurstParams** burst, int port_id);
  virtual Status get_rx_burst(BurstParams** burst);
  virtual void free_rx_metadata(BurstParams* burst) = 0;
  virtual void free_tx_metadata(BurstParams* burst) = 0;
  virtual Status get_tx_metadata_buffer(BurstParams** burst) = 0;
  virtual Status send_tx_burst(BurstParams* burst) = 0;
  virtual Status get_mac_addr(int port, char* mac) = 0;
  virtual int get_port_id(const std::string& key) final;  // NOLINT(readability/inheritance)
  virtual bool validate_config() const;
  virtual uint16_t get_num_rx_queues(int port_id) const;

  virtual Status rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port,
                                        uintptr_t* conn_id);
  virtual Status rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port,
                                        const std::string& src_addr, uintptr_t* conn_id);
  virtual Status rdma_get_port_queue(uintptr_t conn_id, uint16_t* port, uint16_t* queue);
  virtual Status rdma_get_server_conn_id(const std::string& server_addr, uint16_t server_port,
                                         uintptr_t* conn_id);
  virtual Status rdma_set_header(BurstParams* burst, RDMAOpCode op_code, uintptr_t conn_id,
                                 bool is_server, int num_pkts, uint64_t wr_id,
                                 const std::string& local_mr_name);
  virtual RDMAOpCode rdma_get_opcode(BurstParams* burst);
  int numa_from_mem(const MemoryRegionConfig& mr) const;
  Status register_memory_regions();
  Status map_memory_regions();
  struct rte_mempool* create_pktmbuf_pool(const std::string& name, const MemoryRegionConfig& mr);
  struct rte_mempool* create_generic_pool(const std::string& name, const MemoryRegionConfig& mr);

  virtual ~Manager() = default;

 protected:
  static constexpr int MAX_IFS = 4;
  static constexpr int MAX_GPUS = 8;
  static constexpr uint32_t GPU_PAGE_SHIFT = 16;
  static constexpr uint32_t GPU_PAGE_SIZE = (1UL << GPU_PAGE_SHIFT);
  static constexpr uint32_t JUMBO_FRAME_MAX_SIZE = 0x2600;
  static constexpr uint32_t NON_JUMBO_FRAME_MAX_SIZE = 1518;
  bool initialized_ = false;
  NetworkConfig cfg_;
  std::unordered_map<std::string, AllocRegion> ar_;
  std::unordered_map<std::string, std::shared_ptr<struct rte_pktmbuf_extmem>> ext_pktmbufs_;
  std::unordered_map<uint32_t, std::vector<std::pair<uint16_t, uint16_t>>> rx_core_q_map;

  // State for round-robin burst retrieval
  size_t next_port_index_ = 0;                            // For get_rx_burst next port check
  std::unordered_map<int, size_t> next_queue_index_map_;  // For get_rx_burst next queue check

  virtual Status allocate_memory_regions();
  virtual void adjust_memory_regions() {}
  void init_rx_core_q_map();
  size_t get_alignment(MemoryKind kind);
  Status populate_pool(struct rte_ring* ring, const std::string& mr_name);
};

class ManagerFactory {
 public:
  static void set_manager_type(ManagerType type) {
    if (ManagerType_ != ManagerType::UNKNOWN && ManagerType_ != type) {
      throw std::logic_error("Manager type is already set with another manager type.");
    }
    if (type == ManagerType::DEFAULT) {
      ManagerType_ = get_default_manager_type();
    } else {
      ManagerType_ = type;
    }
  }

  static ManagerType get_manager_type() { return ManagerType_; }

  template <typename Config>
  static ManagerType get_manager_type(const Config& config);

  static ManagerType get_default_manager_type();

  static Manager& get_active_manager() {
    if (ManagerType_ == ManagerType::UNKNOWN) { throw std::logic_error("ManagerType not set"); }
    if (!ManagerInstance_) { ManagerInstance_ = create_instance(ManagerType_); }
    return *ManagerInstance_;
  }

 private:
  ManagerFactory() = default;
  ~ManagerFactory() = default;
  ManagerFactory(const ManagerFactory&) = delete;
  ManagerFactory& operator=(const ManagerFactory&) = delete;

  static std::unique_ptr<Manager> ManagerInstance_;
  static ManagerType ManagerType_;

  static std::unique_ptr<Manager> create_instance(ManagerType type);
};

};  // namespace holoscan::advanced_network
