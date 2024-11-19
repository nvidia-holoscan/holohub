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

#include "adv_network_types.h"
#include <optional>

namespace holoscan::ops {

struct AllocRegion {
  std::string mr_name_;
  void* ptr_;
};

/**
 * @brief (Almost) ABC representing an interface into an ANO backend implementation
 *
 */
class ANOMgr {
 public:
  virtual void initialize() = 0;
  virtual bool is_initialized() const { return initialized_; }
  virtual bool set_config_and_initialize(const AdvNetConfigYaml& cfg) = 0;
  virtual void run() = 0;

  // Common free functions to override
  virtual void* get_pkt_ptr(AdvNetBurstParams* burst, int idx) = 0;
  virtual uint16_t get_pkt_len(AdvNetBurstParams* burst, int idx) = 0;
  virtual void* get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) = 0;
  virtual uint16_t get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) = 0;
  virtual uint16_t get_pkt_flow_id(AdvNetBurstParams* burst, int idx) = 0;
  virtual void* get_pkt_extra_info(AdvNetBurstParams* burst, int idx) = 0;
  virtual AdvNetStatus get_tx_pkt_burst(AdvNetBurstParams* burst) = 0;
  virtual AdvNetStatus set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) = 0;
  virtual AdvNetStatus set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                                    unsigned int src_host, unsigned int dst_host) = 0;
  virtual AdvNetStatus set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len,
                                   uint16_t src_port, uint16_t dst_port) = 0;
  virtual AdvNetStatus set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) = 0;
  virtual bool tx_burst_available(AdvNetBurstParams* burst) = 0;

  virtual AdvNetStatus set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                    const std::initializer_list<int>& lens) = 0;
  virtual void free_all_seg_pkts(AdvNetBurstParams* burst, int seg) = 0;
  virtual void free_all_pkts(AdvNetBurstParams* burst) = 0;
  virtual void free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) = 0;
  virtual void free_pkt(AdvNetBurstParams* burst, int pkt) = 0;
  virtual void free_rx_burst(AdvNetBurstParams* burst) = 0;
  virtual void free_tx_burst(AdvNetBurstParams* burst) = 0;
  virtual AdvNetStatus set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t time) = 0;
  virtual void shutdown() = 0;
  virtual void print_stats() = 0;
  virtual uint64_t get_burst_tot_byte(AdvNetBurstParams* burst) = 0;
  virtual AdvNetBurstParams* create_burst_params() = 0;

  /* Internal functions used by ANO operators */
  virtual std::optional<uint16_t> get_port_from_ifname(const std::string& name) = 0;
  virtual AdvNetStatus get_rx_burst(AdvNetBurstParams** burst) = 0;
  virtual void free_rx_meta(AdvNetBurstParams* burst) = 0;
  virtual void free_tx_meta(AdvNetBurstParams* burst) = 0;
  virtual AdvNetStatus get_tx_meta_buf(AdvNetBurstParams** burst) = 0;
  virtual AdvNetStatus send_tx_burst(AdvNetBurstParams* burst) = 0;
  virtual AdvNetStatus get_mac(int port, char* mac) = 0;
  virtual int address_to_port(const std::string& addr) = 0;
  virtual bool validate_config() const;

  virtual ~ANOMgr() = default;

 protected:
  static constexpr int MAX_IFS = 4;
  static constexpr int MAX_GPUS = 8;
  static constexpr uint32_t GPU_PAGE_SHIFT = 16;
  static constexpr uint32_t GPU_PAGE_SIZE = (1UL << GPU_PAGE_SHIFT);
  static constexpr uint32_t JUMBO_FRAME_MAX_SIZE = 0x2600;
  static constexpr uint32_t NON_JUMBO_FRAME_MAX_SIZE = 1518;
  bool initialized_ = false;
  AdvNetConfigYaml cfg_;
  std::unordered_map<std::string, AllocRegion> ar_;

  virtual AdvNetStatus allocate_memory_regions();
  virtual void adjust_memory_regions() {}
};

class AnoMgrFactory {
 public:
  static void set_manager_type(AnoMgrType type) {
    if (AnoMgrType_ != AnoMgrType::UNKNOWN && AnoMgrType_ != type) {
      throw std::logic_error("Manager type is already set with another manager type.");
    }
    if (type == AnoMgrType::DEFAULT) {
      AnoMgrType_ = get_default_manager_type();
    } else {
      AnoMgrType_ = type;
    }
  }

  static AnoMgrType get_manager_type() { return AnoMgrType_; }

  template <typename Config>
  static AnoMgrType get_manager_type(const Config& config);

  static AnoMgrType get_default_manager_type();

  static ANOMgr& get_active_manager() {
    if (AnoMgrType_ == AnoMgrType::UNKNOWN) { throw std::logic_error("AnoMgrType not set"); }
    if (!AnoMgrInstance_) { AnoMgrInstance_ = create_instance(AnoMgrType_); }
    return *AnoMgrInstance_;
  }

 private:
  AnoMgrFactory() = default;
  ~AnoMgrFactory() = default;
  AnoMgrFactory(const AnoMgrFactory&) = delete;
  AnoMgrFactory& operator=(const AnoMgrFactory&) = delete;

  static std::unique_ptr<ANOMgr> AnoMgrInstance_;
  static AnoMgrType AnoMgrType_;

  static std::unique_ptr<ANOMgr> create_instance(AnoMgrType type);
};

};  // namespace holoscan::ops
