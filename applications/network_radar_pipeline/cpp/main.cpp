/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common.h"
#include "source.h"
#include "process.h"
#include "daqiri_connectors/adv_networking_tx.h"
#include "daqiri_connectors/adv_networking_rx.h"
#include "holoscan/holoscan.hpp"
#include <daqiri/daqiri.h>

class App : public holoscan::Application {
 private:
  bool use_daqiri(const std::string& section) {
    bool enabled = true;

    for (const auto& yaml_node : config().yaml_nodes()) {
      const auto params = yaml_node[section];
      if (!params || !params.IsMap()) { continue; }

      if (params["use_daqiri"].IsDefined()) { enabled = params["use_daqiri"].as<bool>(); }
    }

    if (!enabled) {
      HOLOSCAN_LOG_ERROR("Only DAQIRI networking is supported by this application");
      exit(1);
    }

    return enabled;
  }

  /**
   * @brief Setup the application as a radar I/Q data generation pipeline
   */
  void setup_tx() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Initializing radar pipeline as data source");

    // Target/signal simulation
    auto target_sim = make_operator<ops::TargetSimulator>(
      "target_sim",
      from_config("radar_pipeline"));

    if (use_daqiri("tx_params")) {
      auto adv_packet_gen = make_operator<ops::AdvConnectorOpTx>(
        "packet_gen",
        from_config("daqiri"),
        from_config("radar_pipeline"),
        from_config("tx_params"),
        make_condition<BooleanCondition>("is_alive", true));

      add_flow(target_sim, adv_packet_gen, {{"rf_out", "rf_in"}});
    }
  }

  /**
   * @brief Setup the application as a radar signal processing pipeline
   */
  void setup_rx() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Initializing radar pipeline as data processor");

    // Radar algorithms
    auto pc   = make_operator<ops::PulseCompressionOp>(
      "pulse_compression",
      from_config("radar_pipeline"),
      make_condition<CountCondition>(from_config("radar_pipeline.num_transmits").as<size_t>()));
    auto tpc  = make_operator<ops::ThreePulseCancellerOp>(
      "three_pulse_canceller",
      from_config("radar_pipeline"));
    auto dop  = make_operator<ops::DopplerOp>("doppler", from_config("radar_pipeline"));
    auto cfar = make_operator<ops::CFAROp>("cfar", from_config("radar_pipeline"));

    if (use_daqiri("rx_params")) {
      auto adv_rx_pkt = make_operator<ops::AdvConnectorOpRx>(
        "bench_rx",
        from_config("rx_params"),
        from_config("radar_pipeline"),
        make_condition<BooleanCondition>("is_alive", true));
      add_flow(adv_rx_pkt, pc,         {{"rf_out", "rf_in"}});
    }

    add_flow(pc, tpc,   {{"pc_out", "tpc_in"}});
    add_flow(tpc, dop,  {{"tpc_out", "dop_in"}});
    add_flow(dop, cfar, {{"dop_out", "cfar_in"}});
  }

 public:
  void compose() {
    using namespace holoscan;

    if (from_config("radar_pipeline.is_source").as<bool>()) {
      setup_tx();
    } else {
      setup_rx();
    }
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} [source.yaml, process.yaml]", argv[0]);
    return -1;
  }
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/" + std::string(argv[1]);

  // Check if the file exists
  if (!std::filesystem::exists(config_path)) {
    HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist",
                       static_cast<std::string>(config_path));
    return -1;
  }

  if (daqiri::daqiri_init(config_path.string()) != daqiri::Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to configure DAQIRI");
    return -1;
  }
  HOLOSCAN_LOG_INFO("Configured DAQIRI");

  // Run
  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", app->from_config("scheduler")));
  app->run();

  daqiri::print_stats();
  daqiri::shutdown();
  return 0;
}
