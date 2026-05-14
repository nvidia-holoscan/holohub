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
#if DAQIRI_MGR_DPDK
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

    const auto [rx_en, tx_en] = daqiri::get_rx_tx_configs_enabled(config());
    const auto mgr_type = daqiri::get_manager_type();

    HOLOSCAN_LOG_INFO("Using DAQIRI manager {}", daqiri::manager_type_to_string(mgr_type));

    if (mgr_type == daqiri::ManagerType::DPDK) {
#if DAQIRI_MGR_DPDK
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
#else
      HOLOSCAN_LOG_ERROR("DPDK manager/backend is disabled");
      exit(1);
#endif

    } else {
      HOLOSCAN_LOG_ERROR("Unsupported DAQIRI manager/backend");
      exit(1);
    }
  }
};

int main(int argc, char** argv) {
  using namespace holoscan;
  auto app = make_application<App>();

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
    return -1;
  }

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
