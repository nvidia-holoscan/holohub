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

#include <string>
#include <yaml-cpp/yaml.h>

#include "advanced_network/manager.h"

namespace holoscan::advanced_network {

class RmaxMgr : public Manager {
 public:
  RmaxMgr();
  ~RmaxMgr();
  bool set_config_and_initialize(const NetworkConfig& cfg) override;
  void initialize() override;
  void run() override;

  static Status parse_rx_queue_rivermax_config(const YAML::Node& q_item, RxQueueConfig& q);
  static Status parse_tx_queue_rivermax_config(const YAML::Node& q_item, TxQueueConfig& q);

  void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx) override;
  void* get_packet_ptr(BurstParams* burst, int idx) override;
  uint32_t get_segment_packet_length(BurstParams* burst, int seg, int idx) override;
  uint32_t get_packet_length(BurstParams* burst, int idx) override;
  uint16_t get_packet_flow_id(BurstParams* burst, int idx) override;
  void* get_packet_extra_info(BurstParams* burst, int idx) override;
  Status get_tx_packet_burst(BurstParams* burst) override;
  Status set_eth_header(BurstParams* burst, int idx, char* dst_addr) override;
  Status set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                            unsigned int src_host, unsigned int dst_host) override;
  Status set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                           uint16_t dst_port) override;
  Status set_udp_payload(BurstParams* burst, int idx, void* data, int len) override;
  bool is_tx_burst_available(BurstParams* burst) override;

  Status set_packet_lengths(BurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
  void free_all_segment_packets(BurstParams* burst, int seg) override;
  void free_all_packets(BurstParams* burst) override;
  void free_packet_segment(BurstParams* burst, int seg, int pkt) override;
  void free_packet(BurstParams* burst, int pkt) override;
  void free_rx_burst(BurstParams* burst) override;
  void free_tx_burst(BurstParams* burst) override;

  Status get_rx_burst(BurstParams** burst, int port, int q) override;
  using holoscan::advanced_network::Manager::get_rx_burst;  // for overloads
  Status set_packet_tx_time(BurstParams* burst, int idx, uint64_t timestamp);
  void free_rx_metadata(BurstParams* burst) override;
  void free_tx_metadata(BurstParams* burst) override;
  Status get_tx_metadata_buffer(BurstParams** burst) override;
  Status send_tx_burst(BurstParams* burst) override;
  void shutdown() override;
  void print_stats() override;
  uint64_t get_burst_tot_byte(BurstParams* burst) override;
  BurstParams* create_tx_burst_params() override;
  Status get_mac_addr(int port, char* mac) override;

 private:
  class RmaxMgrImpl;
  std::unique_ptr<RmaxMgr::RmaxMgrImpl> pImpl;
};

};  // namespace holoscan::advanced_network
