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

#include "adv_network_rx.h"
#include "adv_network_mgr.h"
#include <memory>

namespace holoscan::ops {

struct AdvNetworkOpRx::AdvNetworkOpRxImpl {
  ANOMgr *ano_mgr;
  struct rte_ring *rx_ring;
  struct rte_mempool *rx_desc_pool;
  struct rte_mempool *rx_meta_pool;
  AdvNetConfigYaml cfg;
};


void AdvNetworkOpRx::setup(OperatorSpec& spec) {
  spec.output<std::shared_ptr<AdvNetBurstParams>>("bench_rx_out");

  spec.param(
      cfg_,
      "cfg",
      "Configuration",
      "Configuration for the advanced network operator",
      AdvNetConfigYaml());
}

void AdvNetworkOpRx::initialize() {
  HOLOSCAN_LOG_INFO("AdvNetworkOpRx::initialize()");
  register_converter<holoscan::ops::AdvNetConfigYaml>();

  holoscan::Operator::initialize();
  Init();
}

int AdvNetworkOpRx::Init() {
  impl = new AdvNetworkOpRxImpl();
  impl->cfg = cfg_.get();
  impl->dpdk_mgr = &dpdk_mgr;
  impl->dpdk_mgr->SetConfigAndInitialize(impl->cfg);
  impl->rx_desc_pool = nullptr;
  impl->rx_ring = nullptr;

  for (const auto &rx : impl->cfg.rx_) {
    auto port_opt = adv_net_get_port_from_ifname(rx.if_name_);
    if (!port_opt.has_value()) {
      HOLOSCAN_LOG_ERROR("Failed to get port from name {}", rx.if_name_);
      return -1;
    }

    for (const auto &q : rx.queues_) {
      pq_map_[(port_opt.value() << 16) | q.common_.id_] = q.output_port_;
    }
  }

  return 0;
}



void AdvNetworkOpRx::compute([[maybe_unused]] InputContext&, OutputContext& op_output,
      [[maybe_unused]] ExecutionContext&) {
  int n;
  AdvNetBurstParams *burst;

  if (unlikely(impl->rx_ring == nullptr)) {
    impl->rx_ring = rte_ring_lookup("RX_RING");
  }

  if (unlikely(impl->rx_meta_pool == nullptr)) {
    impl->rx_meta_pool = rte_mempool_lookup("RX_META_POOL");
  }

  if (rte_ring_dequeue(impl->rx_ring, reinterpret_cast<void**>(&burst)) < 0) {
    return;
  }

  auto adv_burst = std::make_shared<AdvNetBurstParams>();
  memcpy(adv_burst.get(), burst, sizeof(*burst));
  rte_mempool_put(impl->rx_meta_pool, burst);

  const auto port_str = pq_map_[(adv_burst->hdr.hdr.port_id << 16) | adv_burst->hdr.hdr.q_id];
  op_output.emit(adv_burst, port_str.c_str());
}

};  // namespace holoscan::ops
