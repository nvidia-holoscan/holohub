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
#include "adv_network_mgr.h"
#include <memory>

namespace holoscan::ops {

extern ANOMgr *g_ano_mgr;

struct AdvNetworkOpTx::AdvNetworkOpTxImpl {
  AdvNetConfigYaml cfg;
};


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
    throw std::runtime_error("ANO initialization failed");
  }
}

int AdvNetworkOpTx::Init() {
  impl = new AdvNetworkOpTxImpl();
  impl->cfg = cfg_.get();
  set_ano_mgr(impl->cfg);

  if (!g_ano_mgr->set_config_and_initialize(impl->cfg)) {
    return -1;
  }

  return 0;
}

void AdvNetworkOpTx::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
      [[maybe_unused]] ExecutionContext&) {
  int n;

  AdvNetBurstParams *d_params;
  auto rx = op_input.receive<AdvNetBurstParams *>("burst_in");

  if (rx.has_value() && rx.value() != nullptr) {
    const auto tx_buf_res = g_ano_mgr->get_tx_meta_buf(&d_params);
    if (tx_buf_res != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_CRITICAL("Failed to get TX meta descriptor: {}", static_cast<int>(tx_buf_res));
      return;
    }

    AdvNetBurstParams *burst = rx.value();
    memcpy(static_cast<void*>(d_params), burst, sizeof(*burst));

    const auto tx_res = g_ano_mgr->send_tx_burst(d_params);
    if (tx_res != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to send TX burst to ANO: {}", static_cast<int>(tx_res));
      return;
    }

    if (impl->cfg.common_.mgr_ != "doca")
      delete burst;
  }
}
};  // namespace holoscan::ops
