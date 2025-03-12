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
#if ANO_MGR_DPDK || ANO_MGR_RIVERMAX
#include "default_bench_op_rx.h"
#include "default_bench_op_tx.h"
#endif
#if ANO_MGR_GPUNETIO
#include "doca_bench_op_rx.h"
#include "doca_bench_op_tx.h"
#endif
#include "advanced_network/kernels.h"
#include "holoscan/holoscan.hpp"
#include <assert.h>
#include <sys/time.h>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    
    if (holoscan::ops::adv_net_init(config()) != holoscan::ops::AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to initialize advanced network");
      exit(1);
    }

    HOLOSCAN_LOG_INFO("Initializing advanced network operator");
    const auto [rx_en, tx_en] = advanced_network::get_rx_tx_configs_enabled(config());
    const auto mgr_type = advanced_network::get_manager_type(config());
    auto output_rx_ports = advanced_network::get_port_names(config(), "rx");

    HOLOSCAN_LOG_INFO("Using ANO manager {}", advanced_network::manager_type_to_string(mgr_type));

    // DPDK is the default manager backend
    if (mgr_type == advanced_network::ManagerType::DPDK) {
#if ANO_MGR_DPDK
      if (rx_en) {
        auto bench_rx =
            make_operator<ops::AdvNetworkingBenchDefaultRxOp>("bench_rx", from_config("bench_rx"),
            make_condition<BooleanCondition>("is_alive", true));
      }
      if (tx_en) {
        auto bench_tx = make_operator<ops::AdvNetworkingBenchDefaultTxOp>(
            "bench_tx",
            from_config("bench_tx"),
            make_condition<BooleanCondition>("is_alive", true));
      }
#else
      HOLOSCAN_LOG_ERROR("DPDK ANO manager/backend is disabled");
      exit(1);
#endif

    } else if (mgr_type == advanced_network::ManagerType::DOCA) {
#if ANO_MGR_GPUNETIO
      if (rx_en) {
        auto bench_rx =
            make_operator<ops::AdvNetworkingBenchDocaRxOp>("bench_rx", from_config("bench_rx"),
            make_condition<BooleanCondition>("is_alive", true));
      }
      if (tx_en) {
        auto bench_tx = make_operator<ops::AdvNetworkingBenchDocaTxOp>(
            "bench_tx",
            from_config("bench_tx"),
            make_condition<BooleanCondition>("is_alive", true));
      }
#else
      HOLOSCAN_LOG_ERROR("DOCA ANO manager/backend is disabled");
      exit(1);
#endif
    } else if (mgr_type == advanced_network::ManagerType::RIVERMAX) {
#if ANO_MGR_RIVERMAX
      if (rx_en) {
        int index = 0;
        for (const auto& port : output_rx_ports) {
          std::string bench_rx_name = "bench_rx_" + std::to_string(index++);
          auto bench_rx = make_operator<ops::AdvNetworkingBenchDefaultRxOp>(bench_rx_name,
                                                                         from_config("bench_rx"));
        }
      }
      if (tx_en) {
        HOLOSCAN_LOG_ERROR("RIVERMAX ANO manager/backend doesn't support TX");
        exit(1);
      }
#else
      HOLOSCAN_LOG_ERROR("RIVERMAX ANO manager/backend is not supported");
      exit(1);
#endif
    } else {
      HOLOSCAN_LOG_ERROR("Invalid ANO manager/backend");
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
  app->config(config_path);
  app->scheduler(app->make_scheduler<MultiThreadScheduler>(
      "multithread-scheduler", app->from_config("scheduler")));
  app->run();

  advanced_network::shutdown();
  return 0;
}
