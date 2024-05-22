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

extern ANOMgr* g_ano_mgr;

/**
 * @brief Structure for passing packets to/from advanced network operator
 *
 * AdvNetBurstParams is populated by the RX advanced network operator before arriving at the user's
 * operator, and the user populates it prior to sending to the TX advanced network operator. The
 * structure describes metadata about a packet batch and its packet pointers.
 *
 */

AdvNetBurstParams* adv_net_create_burst_params() {
  return g_ano_mgr->create_burst_params();
}

void adv_net_free_pkt(AdvNetBurstParams* burst, int pkt) {
  g_ano_mgr->free_pkt(burst, pkt);
}

void adv_net_free_pkt(std::shared_ptr<AdvNetBurstParams> burst, int pkt) {
  adv_net_free_pkt(burst.get(), pkt);
}

void adv_net_free_pkt_seg(AdvNetBurstParams* burst, int seg, int pkt) {
  g_ano_mgr->free_pkt_seg(burst, seg, pkt);
}

void adv_net_free_pkt_seg(std::shared_ptr<AdvNetBurstParams> burst, int seg, int pkt) {
  adv_net_free_pkt_seg(burst.get(), seg, pkt);
}

uint16_t adv_net_get_pkt_len(AdvNetBurstParams* burst, int idx) {
  return g_ano_mgr->get_pkt_len(burst, idx);
}

uint64_t adv_net_get_burst_tot_byte(std::shared_ptr<AdvNetBurstParams> burst) {
  return g_ano_mgr->get_burst_tot_byte(burst.get());
}

uint16_t adv_net_get_pkt_len(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_pkt_len(burst.get(), idx);
}

uint16_t adv_net_get_seg_pkt_len(AdvNetBurstParams* burst, int seg, int idx) {
  return g_ano_mgr->get_seg_pkt_len(burst, seg, idx);
}

uint16_t adv_net_get_seg_pkt_len(std::shared_ptr<AdvNetBurstParams> burst, int seg, int idx) {
  return adv_net_get_seg_pkt_len(burst.get(), seg, idx);
}

void adv_net_free_all_seg_pkts(AdvNetBurstParams* burst, int seg) {
  g_ano_mgr->free_all_seg_pkts(burst, seg);
}

void adv_net_free_all_seg_pkts(std::shared_ptr<AdvNetBurstParams> burst, int seg) {
  return adv_net_free_all_seg_pkts(burst.get(), seg);
}

void adv_net_free_all_burst_pkts(AdvNetBurstParams* burst) {
  g_ano_mgr->free_all_pkts(burst);
}

void adv_net_free_all_burst_pkts(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_all_burst_pkts(burst.get());
}

void adv_net_free_all_pkts_and_burst(AdvNetBurstParams* burst) {
  adv_net_free_all_burst_pkts(burst);
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_free_all_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  adv_net_free_all_pkts_and_burst(burst.get());
}

void adv_net_free_seg_pkts_and_burst(AdvNetBurstParams* burst, int seg) {
  g_ano_mgr->free_all_seg_pkts(burst, seg);
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_free_seg_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst, int seg) {
  adv_net_free_seg_pkts_and_burst(burst.get(), seg);
}

void adv_net_format_eth_addr(char* dst, std::string addr) {
  std::istringstream iss(addr);
  std::string byteString;

  uint8_t byte_cnt = 0;
  while (std::getline(iss, byteString, ':')) {
    if (byteString.length() == 2) {
      uint16_t byte = std::stoi(byteString, nullptr, 16);
      dst[byte_cnt++] = static_cast<char>(byte);
    } else {
      HOLOSCAN_LOG_ERROR("Invalid MAC address format: {}", addr);
      dst[0] = 0x00;
    }
  }
}

AdvNetStatus adv_net_get_mac(int port, char* mac) {
  return g_ano_mgr->get_mac(port, mac);
}

bool adv_net_tx_burst_available(AdvNetBurstParams* burst) {
  return g_ano_mgr->tx_burst_available(burst);
}

bool adv_net_tx_burst_available(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_tx_burst_available(burst.get());
}

int adv_net_address_to_port(const std::string& addr) {
  return g_ano_mgr->address_to_port(addr);
}

AdvNetStatus adv_net_get_tx_pkt_burst(AdvNetBurstParams* burst) {
  if (!g_ano_mgr->tx_burst_available(burst)) return AdvNetStatus::NO_FREE_BURST_BUFFERS;
  return g_ano_mgr->get_tx_pkt_burst(burst);
}

AdvNetStatus adv_net_get_tx_pkt_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_get_tx_pkt_burst(burst.get());
}

AdvNetStatus adv_net_set_eth_hdr(AdvNetBurstParams* burst, int idx, char* dst_addr) {
  return g_ano_mgr->set_eth_hdr(burst, idx, dst_addr);
}

AdvNetStatus adv_net_set_eth_hdr(std::shared_ptr<AdvNetBurstParams> burst, int idx,
                                 char* dst_addr) {
  return adv_net_set_eth_hdr(burst.get(), idx, dst_addr);
}

AdvNetStatus adv_net_set_ipv4_hdr(AdvNetBurstParams* burst, int idx, int ip_len, uint8_t proto,
                                  unsigned int src_host, unsigned int dst_host) {
  return g_ano_mgr->set_ipv4_hdr(burst, idx, ip_len, proto, src_host, dst_host);
}

AdvNetStatus adv_net_set_ipv4_hdr(std::shared_ptr<AdvNetBurstParams> burst, int idx, int ip_len,
                                  uint8_t proto, unsigned int src_host, unsigned int dst_host) {
  return adv_net_set_ipv4_hdr(burst.get(), idx, ip_len, proto, src_host, dst_host);
}

AdvNetStatus adv_net_set_udp_hdr(AdvNetBurstParams* burst, int idx, int udp_len, uint16_t src_port,
                                 uint16_t dst_port) {
  return g_ano_mgr->set_udp_hdr(burst, idx, udp_len, src_port, dst_port);
}

AdvNetStatus adv_net_set_udp_hdr(std::shared_ptr<AdvNetBurstParams> burst, int idx, int udp_len,
                                 uint16_t src_port, uint16_t dst_port) {
  return adv_net_set_udp_hdr(burst.get(), idx, udp_len, src_port, dst_port);
}

AdvNetStatus adv_net_set_udp_payload(AdvNetBurstParams* burst, int idx, void* data, int len) {
  return g_ano_mgr->set_udp_payload(burst, idx, data, len);
}

AdvNetStatus adv_net_set_udp_payload(std::shared_ptr<AdvNetBurstParams> burst, int idx, void* data,
                                     int len) {
  return adv_net_set_udp_payload(burst.get(), idx, data, len);
}

AdvNetStatus adv_net_set_pkt_lens(AdvNetBurstParams* burst, int idx,
                                  const std::initializer_list<int>& lens) {
  return g_ano_mgr->set_pkt_lens(burst, idx, lens);
}

AdvNetStatus adv_net_set_pkt_lens(std::shared_ptr<AdvNetBurstParams> burst, int idx,
                                  const std::initializer_list<int>& lens) {
  return adv_net_set_pkt_lens(burst.get(), idx, lens);
}

AdvNetStatus adv_net_set_pkt_tx_time(AdvNetBurstParams* burst, int idx, uint64_t time) {
  return g_ano_mgr->set_pkt_tx_time(burst, idx, time);
}

AdvNetStatus adv_net_set_pkt_tx_time(std::shared_ptr<AdvNetBurstParams> burst, int idx,
                                     uint64_t time) {
  return adv_net_set_pkt_tx_time(burst.get(), idx, time);
}

int64_t adv_net_get_num_pkts(AdvNetBurstParams* burst) {
  return burst->hdr.hdr.num_pkts;
}

int64_t adv_net_get_num_pkts(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_get_num_pkts(burst.get());
}

int64_t adv_net_get_q_id(AdvNetBurstParams* burst) {
  return burst->hdr.hdr.q_id;
}

int64_t adv_net_get_q_id(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_get_q_id(burst.get());
}

void adv_net_set_num_pkts(AdvNetBurstParams* burst, int64_t num) {
  burst->hdr.hdr.num_pkts = num;
}

void adv_net_set_num_pkts(std::shared_ptr<AdvNetBurstParams> burst, int64_t num) {
  return adv_net_set_num_pkts(burst.get(), num);
}

void adv_net_set_hdr(AdvNetBurstParams* burst, uint16_t port, uint16_t q, int64_t num, int segs) {
  burst->hdr.hdr.num_pkts = num;
  burst->hdr.hdr.port_id = port;
  burst->hdr.hdr.q_id = q;
  burst->hdr.hdr.num_segs = segs;
}

void adv_net_set_hdr(std::shared_ptr<AdvNetBurstParams> burst, uint16_t port, uint16_t q,
                     int64_t num, int segs) {
  return adv_net_set_hdr(burst.get(), port, q, num, segs);
}

void adv_net_free_tx_burst(AdvNetBurstParams* burst) {
  g_ano_mgr->free_tx_burst(burst);
}

void adv_net_free_tx_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_tx_burst(burst.get());
}

void adv_net_free_rx_burst(AdvNetBurstParams* burst) {
  g_ano_mgr->free_rx_burst(burst);
}

void adv_net_free_rx_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_rx_burst(burst.get());
}

void* adv_net_get_seg_pkt_ptr(AdvNetBurstParams* burst, int seg, int idx) {
  return g_ano_mgr->get_seg_pkt_ptr(burst, seg, idx);
}

void* adv_net_get_seg_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int seg, int idx) {
  return adv_net_get_seg_pkt_ptr(burst.get(), seg, idx);
}

void* adv_net_get_pkt_ptr(AdvNetBurstParams* burst, int idx) {
  return g_ano_mgr->get_pkt_ptr(burst, idx);
}

void* adv_net_get_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_pkt_ptr(burst.get(), idx);
}

std::optional<uint16_t> adv_net_get_port_from_ifname(const std::string& name) {
  return g_ano_mgr->get_port_from_ifname(name);
}

void adv_net_shutdown() {
  g_ano_mgr->shutdown();
}

void adv_net_print_stats() {
  g_ano_mgr->print_stats();
}

};  // namespace holoscan::ops
