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

#include "adv_network_mgr.h"
#include "adv_network_common.h"
#include "holoscan/holoscan.hpp"
#include <rte_mbuf.h>
#include <rte_memcpy.h>
#include <rte_ethdev.h>

namespace holoscan::ops {

  extern ANOMgr *g_ano_mgr;

/**
 * @brief Structure for passing packets to/from advanced network operator
 *
 * AdvNetBurstParams is populated by the RX advanced network operator before arriving at the user's
 * operator, and the user populates it prior to sending to the TX advanced network operator. The
 * structure describes metadata about a packet batch and its packet pointers.
 *
 */


AdvNetBurstParams *adv_net_create_burst_params() {
  return new AdvNetBurstParams();
}

void adv_net_free_pkt(void *pkt) {
  g_ano_mgr->free_pkt(pkt);
}

uint16_t adv_net_get_cpu_pkt_len(AdvNetBurstParams *burst, int idx) {
  return g_ano_mgr->get_cpu_pkt_len(burst, idx);
}

uint16_t adv_net_get_cpu_pkt_len(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_cpu_pkt_len(burst.get(), idx);
}

uint16_t adv_net_get_gpu_pkt_len(AdvNetBurstParams *burst, int idx) {
  return g_ano_mgr->get_gpu_pkt_len(burst, idx);
}

uint16_t adv_net_get_gpu_pkt_len(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_gpu_pkt_len(burst.get(), idx);
}

void adv_net_free_pkts(void **pkts, int num_pkts) {
  g_ano_mgr->free_pkts(pkts, num_pkts);
}

void adv_net_free_all_burst_pkts(AdvNetBurstParams *burst) {
  adv_net_free_pkts(burst->cpu_pkts, burst->hdr.hdr.num_pkts);
  adv_net_free_pkts(burst->gpu_pkts, burst->hdr.hdr.num_pkts);
}

void adv_net_free_all_burst_pkts(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_all_burst_pkts(burst.get());
}

void adv_net_free_all_burst_pkts_and_burst(AdvNetBurstParams *burst) {
  if (burst->cpu_pkts != nullptr) {
    g_ano_mgr->free_pkts(burst->cpu_pkts, burst->hdr.hdr.num_pkts);
  }

  if (burst->gpu_pkts != nullptr) {
    g_ano_mgr->free_pkts(burst->gpu_pkts, burst->hdr.hdr.num_pkts);
  }
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_free_all_burst_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  adv_net_free_all_burst_pkts_and_burst(burst.get());
}

void adv_net_free_cpu_pkts_and_burst(AdvNetBurstParams *burst) {
  g_ano_mgr->free_pkts(burst->cpu_pkts, burst->hdr.hdr.num_pkts);
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_free_cpu_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  adv_net_free_cpu_pkts_and_burst(burst.get());
}

void adv_net_format_eth_addr(char *dst, std::string addr) {
  g_ano_mgr->format_eth_addr(dst, addr);
}

bool adv_net_tx_burst_available(AdvNetBurstParams *burst) {
  return g_ano_mgr->tx_burst_available(burst);
}

bool adv_net_tx_burst_available(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_tx_burst_available(burst.get());
}


AdvNetStatus adv_net_get_tx_pkt_burst(AdvNetBurstParams *burst) {
  return g_ano_mgr->get_tx_pkt_burst(burst);
}

AdvNetStatus adv_net_get_tx_pkt_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_get_tx_pkt_burst(burst.get());
}

AdvNetStatus adv_net_set_cpu_eth_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      char *dst_addr) {
  return g_ano_mgr->set_cpu_eth_hdr(burst, idx, dst_addr);
}

AdvNetStatus adv_net_set_cpu_eth_hdr(std::shared_ptr<AdvNetBurstParams> burst,
                                      int idx,
                                      char *dst_addr) {
  return adv_net_set_cpu_eth_hdr(burst.get(), idx, dst_addr);
}

AdvNetStatus adv_net_set_cpu_ipv4_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      int ip_len,
                                      uint8_t proto,
                                      unsigned int src_host,
                                      unsigned int dst_host) {
  return g_ano_mgr->set_cpu_ipv4_hdr(burst, idx, ip_len, proto, src_host, dst_host);
}

AdvNetStatus adv_net_set_cpu_ipv4_hdr(std::shared_ptr<AdvNetBurstParams> burst,
                                      int idx,
                                      int ip_len,
                                      uint8_t proto,
                                      unsigned int src_host,
                                      unsigned int dst_host) {
  return adv_net_set_cpu_ipv4_hdr(burst.get(),
                                      idx,
                                      ip_len,
                                      proto,
                                      src_host,
                                      dst_host);
}

AdvNetStatus adv_net_set_cpu_udp_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port) {
  return g_ano_mgr->set_cpu_udp_hdr(burst, idx, udp_len, src_port, dst_port);
}

AdvNetStatus adv_net_set_cpu_udp_hdr(std::shared_ptr<AdvNetBurstParams> burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port) {
  return adv_net_set_cpu_udp_hdr(burst.get(), idx, udp_len, src_port, dst_port);
}

AdvNetStatus adv_net_set_cpu_udp_payload(AdvNetBurstParams *burst, int idx, void *data, int len) {
  return g_ano_mgr->set_cpu_udp_payload(burst, idx, data, len);
}

AdvNetStatus adv_net_set_cpu_udp_payload(std::shared_ptr<AdvNetBurstParams> burst,
              int idx, void *data, int len) {
  return adv_net_set_cpu_udp_payload(burst.get(), idx, data, len);
}

AdvNetStatus adv_net_set_pkt_len(AdvNetBurstParams *burst, int idx, int cpu_len, int gpu_len) {
  return g_ano_mgr->set_pkt_len(burst, idx, cpu_len, gpu_len);
}

AdvNetStatus adv_net_set_pkt_len(std::shared_ptr<AdvNetBurstParams> burst,
                                  int idx,
                                  int cpu_len,
                                  int gpu_len) {
  return adv_net_set_pkt_len(burst.get(), idx, cpu_len, gpu_len);
}

int64_t adv_net_get_num_pkts(AdvNetBurstParams *burst) {
  return burst->hdr.hdr.num_pkts;
}

int64_t adv_net_get_num_pkts(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_get_num_pkts(burst.get());
}

int64_t adv_net_get_q_id(AdvNetBurstParams *burst) {
  return burst->hdr.hdr.q_id;
}

int64_t adv_net_get_q_id(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_get_q_id(burst.get());
}

void adv_net_set_num_pkts(AdvNetBurstParams *burst, int64_t num) {
  burst->hdr.hdr.num_pkts = num;
}

void adv_net_set_num_pkts(std::shared_ptr<AdvNetBurstParams> burst, int64_t num) {
  return adv_net_set_num_pkts(burst.get(), num);
}

void adv_net_set_hdr(AdvNetBurstParams *burst, uint16_t port, uint16_t q, int64_t num) {
  burst->hdr.hdr.num_pkts = num;
  burst->hdr.hdr.port_id = port;
  burst->hdr.hdr.q_id = q;
}

void adv_net_set_hdr(std::shared_ptr<AdvNetBurstParams> burst,
          uint16_t port, uint16_t q, int64_t num) {
  return adv_net_set_hdr(burst.get(), port, q, num);
}


void adv_net_free_tx_burst(AdvNetBurstParams *burst) {
  g_ano_mgr->free_tx_burst(burst);
}

void adv_net_free_tx_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_tx_burst(burst.get());
}

void adv_net_free_rx_burst(AdvNetBurstParams *burst) {
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_free_rx_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_rx_burst(burst.get());
}

void *adv_net_get_cpu_pkt_ptr(AdvNetBurstParams *burst, int idx)   {
  return g_ano_mgr->get_cpu_pkt_ptr(burst, idx);
}

void *adv_net_get_cpu_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_cpu_pkt_ptr(burst.get(), idx);
}

void *adv_net_get_gpu_pkt_ptr(AdvNetBurstParams *burst, int idx)   {
  return g_ano_mgr->get_gpu_pkt_ptr(burst, idx);
}

void *adv_net_get_gpu_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_gpu_pkt_ptr(burst.get(), idx);
}

std::optional<uint16_t> adv_net_get_port_from_ifname(const std::string &name) {
  return g_ano_mgr->get_port_from_ifname(name);
}



};  // namespace holoscan::ops
