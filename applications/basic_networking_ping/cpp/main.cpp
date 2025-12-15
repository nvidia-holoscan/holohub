/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "basic_network_operator_rx.h"
#include "basic_network_operator_tx.h"
#include "holoscan/holoscan.hpp"

static constexpr int NUM_MSGS = 10;

namespace holoscan::ops {

class BasicNetworkPingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicNetworkPingTxOp)

  BasicNetworkPingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<std::shared_ptr<NetworkOpBurstParams>>("burst_out");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto mem = new uint8_t[sizeof(index_)];
    auto intp = reinterpret_cast<int*>(mem);
    *intp = index_;

    auto value = std::make_shared<NetworkOpBurstParams>(mem, sizeof(*intp), 1);
    HOLOSCAN_LOG_INFO("Ping message sent with value {}", index_);
    op_output.emit(value, "burst_out");

    index_++;
  }

  int index_ = 0;
};

class BasicNetworkPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicNetworkPingRxOp)

  BasicNetworkPingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    auto in = op_input.receive<std::shared_ptr<NetworkOpBurstParams>>("burst_in").value();
    auto val = *reinterpret_cast<int*>(in->data);
    HOLOSCAN_LOG_INFO("Ping message received with value {}", val);

    delete[] in->data;

    if (val == NUM_MSGS - 1) {
      GxfGraphInterrupt(context.context());
    }
  }

 private:
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto& yaml_nodes = config().yaml_nodes();
    bool rx_en = false;
    bool tx_en = false;

    for (const auto& yaml_node : yaml_nodes) {
      try {
        auto tmp = yaml_node["network_rx"].IsMap();
        rx_en = true;
        HOLOSCAN_LOG_INFO("Receive enabled");
      }
      catch (YAML::InvalidNode &e) {}

      try {
        auto tmp = yaml_node["network_tx"].IsMap();
        tx_en = true;
        HOLOSCAN_LOG_INFO("Transmit enabled");
      }
      catch (YAML::InvalidNode &e) {}
    }

    if (tx_en) {
      auto tx =
          make_operator<ops::BasicNetworkPingTxOp>("tx", make_condition<CountCondition>(NUM_MSGS));
      auto net_tx = make_operator<ops::BasicNetworkOpTx>("network_tx", from_config("network_tx"));
      add_flow(tx, net_tx, {{"burst_out", "burst_in"}});
    }

    if (rx_en) {
      auto net_rx = make_operator<ops::BasicNetworkOpRx>(
          "network_rx", from_config("network_rx"), make_condition<BooleanCondition>("is_alive"));
      auto rx = make_operator<ops::BasicNetworkPingRxOp>("rx");

      add_flow(net_rx, rx, {{"burst_out", "burst_in"}});
    }
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  auto config_path = std::string(argv[1]);
  app->config(config_path);

  app->run();

  return 0;
}
