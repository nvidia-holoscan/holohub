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

#include "adv_network_tx.h"
#include "adv_network_dpdk_mgr.h"
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <memory>

namespace holoscan::ops {

struct AdvNetworkOpTx::AdvNetworkOpTxImpl {
  DpdkMgr *dpdk_mgr;
  struct rte_ring *tx_ring;
  struct rte_mempool *tx_meta_pool;
  AdvNetConfigYaml cfg;
};

struct UDPPkt {
  struct rte_ether_hdr eth;
  struct rte_ipv4_hdr ip;
  struct rte_udp_hdr udp;
  uint8_t payload[];
} __attribute__((packed));

void AdvNetworkOpTx::setup(OperatorSpec& spec)  {
  spec.input<AdvNetBurstParams *>("burst_in");

  spec.param(
      cfg_,
      "cfg",
      "Configuration",
      "Configuration for the advanced network operator",
      AdvNetConfigYaml());
}

void AdvNetworkOpTx::initialize() {
  HOLOSCAN_LOG_INFO("AdvNetworkOpTx::initialize()");
  register_converter<holoscan::ops::AdvNetConfigYaml>();

  holoscan::Operator::initialize();
  if (Init() < 0) {
    HOLOSCAN_LOG_ERROR("Failed to initialize ANO TX");
  }
}

int AdvNetworkOpTx::Init() {
  impl = new AdvNetworkOpTxImpl();
  impl->cfg = cfg_.get();
  impl->dpdk_mgr = &dpdk_mgr;
  impl->tx_ring = nullptr;
  impl->dpdk_mgr->SetConfigAndInitialize(impl->cfg);

  // Set up all LUTs for speed
  for (auto &tx : cfg_.get().tx_) {
    auto port_opt = adv_net_get_port_from_ifname(tx.if_name_);
    if (!port_opt) {
      HOLOSCAN_LOG_CRITICAL("Failed to get port ID from interface {}", tx.if_name_);
      return -1;
    }
  }

  return 0;
}

void AdvNetworkOpTx::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
      [[maybe_unused]] ExecutionContext&) {
  int n;

  if (!unlikely(init)) {
    for (const auto &tx : impl->cfg.tx_) {
      auto port_opt = adv_net_get_port_from_ifname(tx.if_name_);
      if (!port_opt.has_value()) {
        HOLOSCAN_LOG_ERROR("Failed to get port from name {}", tx.if_name_);
        return;
      }

      for (const auto &q : tx.queues_) {
        const auto name = "TX_RING_P" +
          std::to_string(port_opt.value()) + "_Q" + std::to_string(q.common_.id_);
        uint32_t key = (port_opt.value() << 16) | q.common_.id_;
        tx_rings_[key] = rte_ring_lookup(name.c_str());
        if (tx_rings_[key] == nullptr) {
          HOLOSCAN_LOG_ERROR("Failed to look up ring for port {} queue {}",
                              port_opt.value(),  q.common_.id_);
          return;
        }
      }

      impl->tx_meta_pool = rte_mempool_lookup("TX_META_POOL");
      init = true;
    }
  }

  AdvNetBurstParams *d_params;
  auto rx = op_input.receive<AdvNetBurstParams *>("burst_in");

  if (rx.has_value() && rx.value() != nullptr) {
    if (rte_mempool_get(impl->tx_meta_pool, reinterpret_cast<void**>(&d_params)) != 0) {
      HOLOSCAN_LOG_CRITICAL("Failed to get TX meta descriptor");
      return;
    }

    AdvNetBurstParams *burst = rx.value();
    rte_memcpy(static_cast<void*>(d_params), burst, sizeof(*burst));
    struct rte_ring *ring = static_cast<struct rte_ring *>
        (tx_rings_[(burst->hdr.hdr.port_id << 16) | burst->hdr.hdr.q_id]);
    if (rte_ring_enqueue(ring, reinterpret_cast<void *>(d_params)) != 0) {
      adv_net_free_tx_burst(burst);
      rte_mempool_put(impl->tx_meta_pool, d_params);
      HOLOSCAN_LOG_CRITICAL("Failed to enqueue TX work");
      return;
    }

    delete burst;
  }
}
};  // namespace holoscan::ops
