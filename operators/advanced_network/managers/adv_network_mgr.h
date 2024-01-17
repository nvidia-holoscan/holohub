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

/**
 * @brief (Almost) ABC representing an interface into an ANO backend implementation
 * 
 */
class ANOMgr {
  public:
    virtual void initialize() = 0;
    virtual bool is_initialized() const { return initialized_; }
    virtual void set_config_and_initialize(const AdvNetConfigYaml &cfg) = 0;
    virtual void run() = 0;

    // Common free functions to override
    virtual void *get_cpu_pkt_ptr(AdvNetBurstParams *burst, int idx) = 0;
    virtual void *get_gpu_pkt_ptr(AdvNetBurstParams *burst, int idx) = 0;
    virtual uint16_t get_cpu_pkt_len(AdvNetBurstParams *burst, int idx) = 0;
    virtual uint16_t get_gpu_pkt_len(AdvNetBurstParams *burst, int idx) = 0;
    virtual AdvNetStatus get_tx_pkt_burst(AdvNetBurstParams *burst) = 0;
    virtual AdvNetStatus set_cpu_eth_hdr(AdvNetBurstParams *burst, int idx,
                                      char *dst_addr) = 0;
    virtual AdvNetStatus set_cpu_ipv4_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      int ip_len,
                                      uint8_t proto,
                                      unsigned int src_host,
                                      unsigned int dst_host) = 0;
    virtual AdvNetStatus set_cpu_udp_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port) = 0;
    virtual AdvNetStatus set_cpu_udp_payload(AdvNetBurstParams *burst, int idx,
                                      void *data, int len) = 0;
    virtual bool tx_burst_available(AdvNetBurstParams *burst) = 0;

    virtual AdvNetStatus set_pkt_len(AdvNetBurstParams *burst, int idx, int cpu_len, int gpu_len) = 0;
    virtual void free_pkt(void *pkt) = 0;
    virtual void free_pkts(void **pkts, int len) = 0;
    virtual void free_rx_burst(AdvNetBurstParams *burst) = 0;
    virtual void free_tx_burst(AdvNetBurstParams *burst) = 0;

    virtual void format_eth_addr(char *dst, std::string addr) = 0;
    virtual std::optional<uint16_t> get_port_from_ifname(const std::string &name) = 0;
    virtual AdvNetStatus get_rx_burst(AdvNetBurstParams **burst) = 0;
    virtual void free_rx_meta(AdvNetBurstParams *burst) = 0;
    virtual void free_tx_meta(AdvNetBurstParams *burst) = 0;
    virtual AdvNetStatus get_tx_meta_buf(AdvNetBurstParams **burst) = 0;
    virtual AdvNetStatus send_tx_burst(AdvNetBurstParams *burst) = 0;

  protected:
    bool initialized_ = false;
};

void set_ano_mgr(const AdvNetConfigYaml &cfg);

}