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
#include <assert.h>

namespace holoscan::ops {

struct AdvNetworkOpRx::AdvNetworkOpRxImpl {
  AdvNetConfigYaml cfg;
  ANOMgr* mgr;
};

void AdvNetworkOpRx::setup(OperatorSpec& spec) {
  if (output_ports.empty()) { output_ports.insert("bench_rx_out"); }

  for (const auto& port : output_ports) {
    spec.output<std::shared_ptr<AdvNetBurstParams>>(port);
    HOLOSCAN_LOG_INFO("Adding output port {}", port);
  }

  spec.param(cfg_,
             "cfg",
             "Configuration",
             "Configuration for the advanced network operator",
             AdvNetConfigYaml());
}

void AdvNetworkOpRx::stop() {
  HOLOSCAN_LOG_INFO("AdvNetworkOpRx::stop()");
  impl->mgr->shutdown();
}

void AdvNetworkOpRx::initialize() {
  HOLOSCAN_LOG_INFO("AdvNetworkOpRx::initialize()");
  register_converter<holoscan::ops::AdvNetConfigYaml>();

  holoscan::Operator::initialize();
  if (Init() < 0) { throw std::runtime_error("ANO initialization failed"); }
}

int AdvNetworkOpRx::Init() {
  impl = new AdvNetworkOpRxImpl();
  impl->cfg = cfg_.get();
  AnoMgrFactory::set_manager_type(impl->cfg.common_.manager_type);

  impl->mgr = &(AnoMgrFactory::get_active_manager());
  assert(impl->mgr != nullptr && "ANO Manager is not initialized");

  if (!impl->mgr->set_config_and_initialize(impl->cfg)) { return -1; }

  for (const auto& intf : impl->cfg.ifs_) {
    const auto& rx = intf.rx_;
    auto port_opt = impl->mgr->get_port_from_ifname(intf.address_);
    if (!port_opt.has_value()) {
      HOLOSCAN_LOG_ERROR("Failed to get port from name {}", intf.address_);
      return -1;
    }

    for (const auto& q : rx.queues_) {
      pq_map_[(port_opt.value() << 16) | q.common_.id_] = q.output_port_;
    }
  }

  return 0;
}

void AdvNetworkOpRx::compute([[maybe_unused]] InputContext&, OutputContext& op_output,
                             [[maybe_unused]] ExecutionContext&) {
  int n;
  AdvNetBurstParams* burst;

  const auto res = impl->mgr->get_rx_burst(&burst);

  if (res != AdvNetStatus::SUCCESS) { return; }

  auto adv_burst = std::make_shared<AdvNetBurstParams>();
  memcpy(adv_burst.get(), burst, sizeof(*burst));

  impl->mgr->free_rx_meta(burst);

  const auto port_str = pq_map_[(adv_burst->hdr.hdr.port_id << 16) | adv_burst->hdr.hdr.q_id];
  op_output.emit(adv_burst, port_str.c_str());
}

};  // namespace holoscan::ops
