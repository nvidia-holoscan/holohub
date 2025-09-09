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
#if ANO_MGR_RDMA
#include "rdma_bench.h"
#endif
#include "advanced_network/kernels.h"
#include "holoscan/holoscan.hpp"
#include <assert.h>
#include <sys/time.h>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
    if (advanced_network::adv_net_init(adv_net_config) != advanced_network::Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
      exit(1);
    }
    HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

    const auto [rdma_server_en, rdma_client_en] =
        holoscan::advanced_network::get_rdma_configs_enabled(config());
    const auto [rx_en, tx_en] = advanced_network::get_rx_tx_configs_enabled(config());
    const auto mgr_type = advanced_network::get_manager_type(config());

    HOLOSCAN_LOG_INFO("Using Advanced Network manager {}",
                      advanced_network::manager_type_to_string(mgr_type));

    // DPDK is the default manager backend
    if (mgr_type == advanced_network::ManagerType::DPDK) {
#if ANO_MGR_DPDK
      if (rx_en) {
        auto bench_rx = make_operator<ops::AdvNetworkingBenchDefaultRxOp>(
            "bench_rx",
            from_config("bench_rx"),
            make_condition<BooleanCondition>("is_alive", true));
        add_operator(bench_rx);
      }
      if (tx_en) {
        auto bench_tx = make_operator<ops::AdvNetworkingBenchDefaultTxOp>(
            "bench_tx",
            from_config("bench_tx"),
            make_condition<BooleanCondition>("is_alive", true));
        add_operator(bench_tx);
      }
#else
      HOLOSCAN_LOG_ERROR("DPDK manager/backend is disabled");
      exit(1);
#endif

    } else if (mgr_type == advanced_network::ManagerType::DOCA) {
#if ANO_MGR_GPUNETIO
      if (rx_en) {
        auto bench_rx = make_operator<ops::AdvNetworkingBenchDocaRxOp>(
            "bench_rx",
            from_config("bench_rx"),
            make_condition<BooleanCondition>("is_alive", true));
        add_operator(bench_rx);
      }
      if (tx_en) {
        auto bench_tx = make_operator<ops::AdvNetworkingBenchDocaTxOp>(
            "bench_tx",
            from_config("bench_tx"),
            make_condition<BooleanCondition>("is_alive", true));
        add_operator(bench_tx);
      }
#else
      HOLOSCAN_LOG_ERROR("GPUNetIO manager/backend is disabled");
      exit(1);
#endif
    } else if (mgr_type == advanced_network::ManagerType::RIVERMAX) {
#if ANO_MGR_RIVERMAX
      if (rx_en) {
        std::string bench_rx_name = "bench_rx";
        auto bench_rx = make_operator<ops::AdvNetworkingBenchDefaultRxOp>(bench_rx_name,
                                                                          from_config("bench_rx"));
        add_operator(bench_rx);
      }
      if (tx_en) {
        HOLOSCAN_LOG_ERROR("RIVERMAX manager/backend doesn't support TX");
        exit(1);
      }
#else
      HOLOSCAN_LOG_ERROR("RIVERMAX manager/backend is not supported");
      exit(1);
#endif
    } else if (mgr_type == holoscan::advanced_network::ManagerType::RDMA) {
#if ANO_MGR_RDMA
      if (rdma_server_en) {
        auto bench_server = make_operator<ops::AdvNetworkingRdmaOp>(
            "rdma_bench_server",
            from_config("rdma_bench_server"),
            make_condition<BooleanCondition>("is_alive", true));
        add_operator(bench_server);
      }

      if (rdma_client_en) {
        auto bench_client = make_operator<ops::AdvNetworkingRdmaOp>(
            "rdma_bench_client",
            from_config("rdma_bench_client"),
            make_condition<BooleanCondition>("is_alive", true));
        add_operator(bench_client);
      }

#else
      HOLOSCAN_LOG_ERROR("RDMA ANO manager/backend is not supported");
      exit(1);
#endif
    } else {
      HOLOSCAN_LOG_ERROR("Invalid Advanced Network manager/backend");
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
  app->scheduler(app->make_scheduler<MultiThreadScheduler>("multithread-scheduler",
                                                           app->from_config("scheduler")));
  app->run();

  advanced_network::shutdown();
  return 0;
}
