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
#if DAQIRI_ENGINE_DPDK
#include "default_bench_op_rx.h"
#include "default_bench_op_tx.h"
#endif
#include "holoscan/holoscan.hpp"
#include <daqiri/daqiri.h>
#include <assert.h>
#include <sys/time.h>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

#if DAQIRI_ENGINE_DPDK
    const auto [rx_en, tx_en] = daqiri::get_rx_tx_configs_enabled(config());

    // The raw Ethernet benchmark only runs on the DPDK engine, so there is no
    // need to query or branch on the engine type at runtime.
    if (rx_en) {
      auto bench_rx = make_operator<ops::DaqiriRawEthernetBenchDefaultRxOp>(
          "bench_rx",
          from_config("bench_rx"),
          make_condition<BooleanCondition>("is_alive", true));
      add_operator(bench_rx);
    }
    if (tx_en) {
      auto bench_tx = make_operator<ops::DaqiriRawEthernetBenchDefaultTxOp>(
          "bench_tx",
          from_config("bench_tx"),
          make_condition<BooleanCondition>("is_alive", true));
      add_operator(bench_tx);
    }
#endif
  }
};

int main(int argc, char** argv) {
  using namespace holoscan;

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
    return -1;
  }

#if !DAQIRI_ENGINE_DPDK
  HOLOSCAN_LOG_ERROR("DPDK engine is disabled; rebuild with -DDAQIRI_ENGINE=dpdk");
  return -1;
#endif

  auto app = make_application<App>();

  std::filesystem::path config_path(argv[1]);
  if (!config_path.is_absolute()) {
    config_path = std::filesystem::canonical(argv[0]).parent_path() / config_path;
  }
  if (daqiri::daqiri_init(config_path.string()) != daqiri::Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to configure DAQIRI");
    return -1;
  }
  HOLOSCAN_LOG_INFO("Configured DAQIRI");
  app->config(config_path);
  app->scheduler(app->make_scheduler<MultiThreadScheduler>("multithread-scheduler",
                                                           app->from_config("scheduler")));
  app->run();

  daqiri::print_stats();
  daqiri::shutdown();
  return 0;
}
