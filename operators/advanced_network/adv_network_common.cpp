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

void adv_net_format_eth_addr(char *dst, std::string addr) {
  rte_ether_unformat_addr(addr.c_str(), reinterpret_cast<struct rte_ether_addr *>(dst));
}

bool adv_net_tx_burst_available(AdvNetBurstParams *burst) {
  const auto append = "_POOL_P" +
                std::to_string(burst->hdr.hdr.port_id) +
                "_Q" +
                std::to_string(burst->hdr.hdr.q_id);

  const auto burst_pool_name = std::string("TX_BURST") + append;
  const auto burst_pool = rte_mempool_lookup(burst_pool_name.c_str());
  if (burst_pool == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to look up burst pool name for port {} queue {}",
      burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
    return false;
  }

  const std::string cpu_pool_str = std::string("TX_CPU") + append;
  auto cpu_pool   = rte_mempool_lookup(cpu_pool_str.c_str());

  const std::string gpu_pool_str = std::string("TX_GPU") + append;
  auto gpu_pool   = rte_mempool_lookup(gpu_pool_str.c_str());


  // Wait for 2x the number of buffers to be available since some may still be in transit
  // by the NIC and this number can decrease
  auto batch = 0;
  if (cpu_pool != nullptr) {
    if (rte_mempool_avail_count(cpu_pool) < burst->hdr.hdr.num_pkts * 2) {
      return false;
    }
  }

  if (gpu_pool != nullptr) {
    if (rte_mempool_avail_count(gpu_pool) < burst->hdr.hdr.num_pkts * 2) {
      return false;
    }
  }

  return true;
}

bool adv_net_tx_burst_available(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_tx_burst_available(burst.get());
}


AdvNetStatus adv_net_get_tx_pkt_burst(AdvNetBurstParams *burst) {
  const auto append = "_POOL_P" +
                std::to_string(burst->hdr.hdr.port_id) +
                "_Q" +
                std::to_string(burst->hdr.hdr.q_id);

  const auto burst_pool_name = std::string("TX_BURST") + append;
  const auto burst_pool = rte_mempool_lookup(burst_pool_name.c_str());
  if (burst_pool == nullptr) {
    HOLOSCAN_LOG_ERROR("Failed to look up burst pool name for port {} queue {}",
      burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
    return AdvNetStatus::NO_FREE_BURST_BUFFERS;;
  }

  const std::string cpu_pool_str = std::string("TX_CPU") + append;
  auto cpu_pool   = rte_mempool_lookup(cpu_pool_str.c_str());

  const std::string gpu_pool_str = std::string("TX_GPU") + append;
  auto gpu_pool   = rte_mempool_lookup(gpu_pool_str.c_str());

  if (cpu_pool) {
    if (rte_mempool_get(burst_pool, reinterpret_cast<void**>(&burst->cpu_pkts)) != 0) {
      return AdvNetStatus::NO_FREE_BURST_BUFFERS;
    }

    if (rte_pktmbuf_alloc_bulk(cpu_pool, reinterpret_cast<rte_mbuf**>(burst->cpu_pkts),
                static_cast<int>(burst->hdr.hdr.num_pkts)) != 0) {
      rte_mempool_put(burst_pool, reinterpret_cast<void*>(burst->cpu_pkts));
      return AdvNetStatus::NO_FREE_CPU_PACKET_BUFFERS;
    }
  }

  if (gpu_pool) {
    if (rte_mempool_get(burst_pool, reinterpret_cast<void**>(&burst->gpu_pkts)) != 0) {
      adv_net_free_pkts(burst->cpu_pkts, burst->hdr.hdr.num_pkts);
      rte_mempool_put(burst_pool, reinterpret_cast<void*>(burst->cpu_pkts));
      return AdvNetStatus::NO_FREE_BURST_BUFFERS;
    }

    if (rte_pktmbuf_alloc_bulk(gpu_pool, reinterpret_cast<rte_mbuf**>(burst->gpu_pkts),
                static_cast<int>(burst->hdr.hdr.num_pkts)) != 0) {
      adv_net_free_pkts(burst->cpu_pkts, burst->hdr.hdr.num_pkts);
      rte_mempool_put(burst_pool, reinterpret_cast<void*>(burst->cpu_pkts));
      rte_mempool_put(burst_pool, reinterpret_cast<void*>(burst->gpu_pkts));
      return AdvNetStatus::NO_FREE_GPU_PACKET_BUFFERS;
    }
  }

  return AdvNetStatus::SUCCESS;
}

AdvNetStatus adv_net_get_tx_pkt_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  return adv_net_get_tx_pkt_burst(burst.get());
}

AdvNetStatus adv_net_set_cpu_eth_hdr(AdvNetBurstParams *burst,
                                      int idx,
                                      char *dst_addr) {
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);
  memcpy(reinterpret_cast<void*>(&mbuf_data->eth.dst_addr),
          reinterpret_cast<void*>(dst_addr),
          sizeof(mbuf_data->eth.dst_addr));

  mbuf_data->eth.ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
  return AdvNetStatus::SUCCESS;
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
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);
  mbuf_data->ip.next_proto_id = proto;
  mbuf_data->ip.ihl = 5;
  mbuf_data->ip.total_length =
        rte_cpu_to_be_16(sizeof(mbuf_data->ip) + ip_len);
  mbuf_data->ip.version = 4;
  mbuf_data->ip.src_addr = htonl(src_host);
  mbuf_data->ip.dst_addr = htonl(dst_host);
  return AdvNetStatus::SUCCESS;
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
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);

  mbuf_data->udp.dgram_cksum = 0;
  mbuf_data->udp.src_port = htons(src_port);
  mbuf_data->udp.dst_port = htons(dst_port);
  mbuf_data->udp.dgram_len = htons(udp_len + sizeof(mbuf_data->udp));
  return AdvNetStatus::SUCCESS;
}

AdvNetStatus adv_net_set_cpu_udp_hdr(std::shared_ptr<AdvNetBurstParams> burst,
                                      int idx,
                                      int udp_len,
                                      uint16_t src_port,
                                      uint16_t dst_port) {
  return adv_net_set_cpu_udp_hdr(burst.get(), idx, udp_len, src_port, dst_port);
}

AdvNetStatus adv_net_set_cpu_udp_payload(AdvNetBurstParams *burst, int idx, void *data, int len) {
  auto mbuf = reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]);
  auto mbuf_data = rte_pktmbuf_mtod(mbuf, UDPPkt*);

  rte_memcpy(mbuf_data->payload, data, len);
  return AdvNetStatus::SUCCESS;
}

AdvNetStatus adv_net_set_cpu_udp_payload(std::shared_ptr<AdvNetBurstParams> burst,
              int idx, void *data, int len) {
  return adv_net_set_cpu_udp_payload(burst.get(), idx, data, len);
}

AdvNetStatus adv_net_set_pkt_len(AdvNetBurstParams *burst, int idx, int cpu_len, int gpu_len) {
  if (cpu_len == 0) {
    auto mbuf = reinterpret_cast<rte_mbuf*>(burst->gpu_pkts[idx]);
    mbuf->data_len = gpu_len;
    mbuf->pkt_len  = gpu_len;
  } else {
    if (gpu_len == 0) {
      auto mbuf = reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]);
      mbuf->data_len = cpu_len;
      mbuf->pkt_len  = cpu_len;
    } else {
      auto cpu_mbuf = reinterpret_cast<rte_mbuf*>(burst->cpu_pkts[idx]);
      auto gpu_mbuf = reinterpret_cast<rte_mbuf*>(burst->gpu_pkts[idx]);
      cpu_mbuf->data_len = cpu_len;
      cpu_mbuf->pkt_len  = cpu_len + gpu_len;
      gpu_mbuf->data_len = gpu_len;
      gpu_mbuf->pkt_len  = cpu_len + gpu_len;
    }
  }

  return AdvNetStatus::SUCCESS;
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
  const auto append = "_POOL_P" +
                std::to_string(burst->hdr.hdr.port_id) +
                "_Q" +
                std::to_string(burst->hdr.hdr.q_id);
  const auto name = "TX_BURST" + append;
  auto burst_pool = rte_mempool_lookup(name.c_str());
  if (burst->cpu_pkts != nullptr) {
    rte_mempool_put(burst_pool, (void *)burst->cpu_pkts);
  }
  if (burst->gpu_pkts != nullptr) {
    rte_mempool_put(burst_pool, (void *)burst->gpu_pkts);
  }
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
