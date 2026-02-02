/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <assert.h>
#include <sys/time.h>
#include <arpa/inet.h>

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include "advanced_network/common.h"
#include "adv_network_media_tx.h"

namespace ano = holoscan::advanced_network;
using holoscan::advanced_network::NetworkConfig;
using holoscan::advanced_network::Status;

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
    if (ano::adv_net_init(adv_net_config) != Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
      exit(1);
    }
    HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

    const auto [rx_en, tx_en] = ano::get_rx_tx_configs_enabled(config());
    const auto mgr_type = ano::get_manager_type(config());

    HOLOSCAN_LOG_INFO("Using Advanced Network manager {}",
                      ano::manager_type_to_string(mgr_type));

    if (!tx_en) {
      HOLOSCAN_LOG_ERROR("Tx is not enabled. Please enable Tx in the config file.");
      exit(1);
    }

    auto adv_net_media_tx = make_operator<ops::AdvNetworkMediaTxOp>(
        "advanced_network_media_tx", from_config("advanced_network_media_tx"));

    ArgList args;
    args.add(Arg("allocator",
                 make_resource<RMMAllocator>("rmm_allocator", from_config("rmm_allocator"))));

    auto replayer =
        make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), args);
    add_flow(replayer, adv_net_media_tx, {{"output", "input"}});
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

  ano::shutdown();
  return 0;
}
