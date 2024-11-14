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

#include "adv_network_mgr.h"

namespace holoscan::ops {

class RmaxMgr : public ANOMgr {
 public:
  RmaxMgr();
  ~RmaxMgr();
  bool set_config_and_initialize(const AdvNetConfigYaml& cfg) override;
  void initialize() override;
  void run() override;

  static AdvNetStatus parse_rx_queue_rivermax_config(const YAML::Node& q_item, RxQueueConfig& q);
  static AdvNetStatus parse_tx_queue_rivermax_config(const YAML::Node& q_item, TxQueueConfig& q);

  void* get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) override;
  void* get_pkt_ptr(AdvNetBurstParams* burst, int idx) override;
  uint16_t get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) override;
  uint16_t get_pkt_len(AdvNetBurstParams* burst, int idx) override;
  uint16_t get_pkt_flow_id(AdvNetBurstParams* burst, int idx) override;
  void* get_pkt_extra_info(AdvNetBurstParams* burst, int idx) override;
  AdvNetStatus get_tx_pkt_burst(AdvNetBurstParams* burst) override;
  AdvNetStatus set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) override;
  AdvNetStatus set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                            unsigned int src_host, unsigned int dst_host) override;
  AdvNetStatus set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                           uint16_t dst_port) override;
  AdvNetStatus set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) override;
  bool tx_burst_available(AdvNetBurstParams* burst) override;

  AdvNetStatus set_pkt_lens(AdvNetBurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
  void free_all_seg_pkts(AdvNetBurstParams* burst, int seg) override;
  void free_all_pkts(AdvNetBurstParams* burst) override;
  void free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) override;
  void free_pkt(AdvNetBurstParams* burst, int pkt) override;
  void free_rx_burst(AdvNetBurstParams* burst) override;
  void free_tx_burst(AdvNetBurstParams* burst) override;

  std::optional<uint16_t> get_port_from_ifname(const std::string& name) override;

  AdvNetStatus get_rx_burst(AdvNetBurstParams** burst) override;
  AdvNetStatus set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t timestamp);
  void free_rx_meta(AdvNetBurstParams* burst) override;
  void free_tx_meta(AdvNetBurstParams* burst) override;
  AdvNetStatus get_tx_meta_buf(AdvNetBurstParams** burst) override;
  AdvNetStatus send_tx_burst(AdvNetBurstParams* burst) override;
  void shutdown() override;
  void print_stats() override;
  uint64_t get_burst_tot_byte(AdvNetBurstParams* burst) override;
  AdvNetBurstParams* create_burst_params() override;
  AdvNetStatus get_mac(int port, char* mac) override;
  int address_to_port(const std::string& addr) override;

 private:
  class RmaxMgrImpl;
  std::unique_ptr<RmaxMgr::RmaxMgrImpl> pImpl;
};

};  // namespace holoscan::ops
