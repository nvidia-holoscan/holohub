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

#include "adv_network_common.h"
#include "holoscan/holoscan.hpp"
#include <rte_mbuf.h>
#include <rte_memcpy.h>
#include <rte_ethdev.h>

namespace holoscan::ops {

/**
 * @brief Structure for passing packets to/from advanced network operator
 *
 * AdvNetBurstParams is populated by the RX advanced network operator before arriving at the user's
 * operator, and the user populates it prior to sending to the TX advanced network operator. The
 * structure describes metadata about a packet batch and its packet pointers.
 *
 */

/**
 * @brief Generic UDP packet structure
 *
 */
struct UDPPkt {
  struct rte_ether_hdr eth;
  struct rte_ipv4_hdr ip;
  struct rte_udp_hdr udp;
  uint8_t payload[];
} __attribute__((packed));


AdvNetBurstParams *adv_net_create_burst_params() {
  return new AdvNetBurstParams();
}

void adv_net_free_pkt(void *pkt) {
  rte_pktmbuf_free_seg(static_cast<rte_mbuf*>(pkt));
}

uint16_t adv_net_get_cpu_packet_len(AdvNetBurstParams *burst, int idx) {
  return reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx])->data_len;
}

uint16_t adv_net_get_cpu_packet_len(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_cpu_packet_len(burst.get(), idx);
}

uint16_t adv_net_get_gpu_packet_len(AdvNetBurstParams *burst, int idx) {
  return reinterpret_cast<rte_mbuf*>(burst->gpu_pkts[idx])->data_len;
}

uint16_t adv_net_get_gpu_packet_len(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return adv_net_get_gpu_packet_len(burst.get(), idx);
}

void adv_net_free_pkts(void **pkts, int num_pkts) {
  for (int p = 0; p < num_pkts; p++) {
    rte_pktmbuf_free_seg(reinterpret_cast<rte_mbuf**>(pkts)[p]);
  }
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
    adv_net_free_pkts(burst->cpu_pkts, burst->hdr.hdr.num_pkts);
  }

  if (burst->gpu_pkts != nullptr) {
    adv_net_free_pkts(burst->gpu_pkts, burst->hdr.hdr.num_pkts);
  }
  adv_net_free_rx_burst(burst);
}

void adv_net_free_all_burst_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  adv_net_free_all_burst_pkts_and_burst(burst.get());
}

void adv_net_free_cpu_pkts_and_burst(AdvNetBurstParams *burst) {
  adv_net_free_pkts(burst->cpu_pkts, burst->hdr.hdr.num_pkts);
  adv_net_free_rx_burst(burst);
}

void adv_net_free_cpu_pkts_and_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  adv_net_free_cpu_pkts_and_burst(burst.get());
}


bool adv_net_tx_burst_available(int num_pkts, int port) {
  auto burst_pool = rte_mempool_lookup("TX_BURST_POOL");

  // Only single queue
  const std::string pool_name = "TX_POOL" + std::string("_P") + std::to_string(port) + "_Q0";
  auto pkt_pool   = rte_mempool_lookup(pool_name.c_str());
  if (burst_pool == nullptr || pkt_pool == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to look up burst pool name for port {}", port);
    return false;
  }

  if (rte_mempool_empty(burst_pool)) {
    return false;
  }

  // Wait for 2x the number of buffers to be available since some may still be in transit
  // by the NIC and this number can decrease
  if (rte_mempool_avail_count(pkt_pool) < num_pkts * 2) {
    return false;
  }

  return true;
}


AdvNetStatus adv_net_get_tx_pkt_burst(AdvNetBurstParams *burst, int port) {
  auto burst_pool = rte_mempool_lookup("TX_BURST_POOL");

  // Only single queue
  const std::string pool_name = "TX_POOL" + std::string("_P") + std::to_string(port) + "_Q0";
  auto pkt_pool   = rte_mempool_lookup(pool_name.c_str());
  if (burst_pool == nullptr || pkt_pool == nullptr) {
    return AdvNetStatus::NULL_PTR;
  }

  if (rte_mempool_get(burst_pool, reinterpret_cast<void**>(&burst->cpu_pkts)) != 0) {
    return AdvNetStatus::NO_FREE_BURST_BUFFERS;
  }

  if (rte_pktmbuf_alloc_bulk(pkt_pool, reinterpret_cast<rte_mbuf**>(burst->cpu_pkts),
              static_cast<int>(burst->hdr.hdr.num_pkts)) != 0) {
    rte_mempool_put(burst_pool, reinterpret_cast<void*>(burst->cpu_pkts));
    return AdvNetStatus::NO_FREE_CPU_PACKET_BUFFERS;
  }

  return AdvNetStatus::SUCCESS;
}

AdvNetStatus adv_net_get_tx_pkt_burst(std::shared_ptr<AdvNetBurstParams> burst, int port) {
  return adv_net_get_tx_pkt_burst(burst.get(), port);
}


AdvNetStatus adv_net_set_cpu_udp_payload(AdvNetBurstParams *burst, int idx, void *data, int len) {
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);

  rte_memcpy(mbuf_data->payload, data, len);
  mbuf_data->eth.ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
  mbuf_data->udp.dgram_cksum = 0;
  mbuf_data->udp.dgram_len = htons(len + sizeof(mbuf_data->udp));
  mbuf_data->ip.next_proto_id = IPPROTO_UDP;
  mbuf_data->ip.ihl = 5;
  mbuf_data->ip.total_length =
        rte_cpu_to_be_16(sizeof(mbuf_data->ip) + sizeof(mbuf_data->udp) + len);
  mbuf_data->ip.version = 4;

  mbuf->data_len = len + sizeof(UDPPkt);
  mbuf->pkt_len  = mbuf->data_len;
  return AdvNetStatus::SUCCESS;
}

AdvNetStatus adv_net_set_cpu_udp_payload(std::shared_ptr<AdvNetBurstParams> burst,
              int idx, void *data, int len) {
  return adv_net_set_cpu_udp_payload(burst.get(), idx, data, len);
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
  auto burst_pool = rte_mempool_lookup("TX_BURST_POOL");
  rte_mempool_put(burst_pool, (void *)burst->cpu_pkts);
}

void adv_net_free_tx_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_tx_burst(burst.get());
}

void adv_net_free_rx_burst(AdvNetBurstParams *burst) {
  auto burst_pool = rte_mempool_lookup("RX_BURST_POOL");
  if (burst->cpu_pkts != nullptr) {
    rte_mempool_put(burst_pool, (void *)burst->cpu_pkts);
  }
  if (burst->gpu_pkts != nullptr) {
    rte_mempool_put(burst_pool, (void *)burst->gpu_pkts);
  }
}

void adv_net_free_rx_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_free_rx_burst(burst.get());
}

void *adv_net_get_cpu_pkt_ptr(AdvNetBurstParams *burst, int idx)   {
  return rte_pktmbuf_mtod(reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]), void*);
}

void *adv_net_get_cpu_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return rte_pktmbuf_mtod(reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]), void*);
}

void *adv_net_get_gpu_pkt_ptr(AdvNetBurstParams *burst, int idx)   {
  return rte_pktmbuf_mtod(reinterpret_cast<rte_mbuf*>(burst->gpu_pkts[idx]), void*);
}

void *adv_net_get_gpu_pkt_ptr(std::shared_ptr<AdvNetBurstParams> burst, int idx) {
  return rte_pktmbuf_mtod(reinterpret_cast<rte_mbuf*>(burst->gpu_pkts[idx]), void*);
}

std::optional<uint16_t> adv_net_get_port_from_ifname(const std::string &name) {
  uint16_t port;
  auto ret = rte_eth_dev_get_port_by_name(name.c_str(), &port);
  if (ret < 0) {
    return {};
  }

  return port;
}



};  // namespace holoscan::ops
