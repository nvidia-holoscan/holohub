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

#pragma once
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <tuple>
#include <stdint.h>
#include <unordered_set>
#include "adv_network_types.h"
#include "holoscan/holoscan.hpp"

template <>
struct YAML::convert<holoscan::ops::AdvNetConfigYaml> {
  static Node encode(const holoscan::ops::AdvNetConfigYaml& input_spec) {
    Node node;
    return node;
  }
  /**
   * @brief Parse flow configuration from a YAML node.
   *
   * @param flow_item The YAML node containing the flow configuration.
   * @param flow The FlowConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_flow_config(const YAML::Node& flow_item, holoscan::ops::FlowConfig& flow);

  /**
   * @brief Parse memory region configuration from a YAML node.
   *
   * @param mr The YAML node containing the memory region configuration.
   * @param tmr The MemoryRegion object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_memory_region_config(const YAML::Node& mr,
                                         holoscan::ops::MemoryRegion& memory_region);

  /**
   * @brief Parse common RX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the RX queue configuration.
   * @param q The RxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_rx_queue_config(const YAML::Node& q_item,
                                    const holoscan::ops::AnoMgrType& manager_type,
                                    holoscan::ops::RxQueueConfig& rx_queue_config);

  /**
   * @brief Parse RX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the RX queue configuration.
   * @param manager_type The manager type.
   * @param q The RxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_rx_queue_common_config(const YAML::Node& q_item,
                                           holoscan::ops::RxQueueConfig& rx_queue_config);

  /**
   * @brief Parse common TX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the TX queue configuration.
   * @param q The TxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_tx_queue_config(const YAML::Node& q_item,
                                    const holoscan::ops::AnoMgrType& manager_type,
                                    holoscan::ops::TxQueueConfig& tx_queue_config);

  /**
   * @brief Parse TX queue configuration from a YAML node.
   *
   * @param q_item The YAML node containing the TX queue configuration.
   * @param manager_type The manager type.
   * @param q The TxQueueConfig object to populate.
   * @return true if parsing was successful, false otherwise.
   */
  static bool parse_tx_queue_common_config(const YAML::Node& q_item,
                                           holoscan::ops::TxQueueConfig& tx_queue_config);

  /**
   * @brief Decode the YAML node into an AdvNetConfigYaml object.
   *
   * This function parses the provided YAML node and populates the given AdvNetConfigYaml object.
   * It handles various configurations such as version, master core, manager type, debug flag,
   * memory regions, interfaces, RX queues, TX queues, and flows.
   *
   * @param node The YAML node containing the configuration.
   * @param input_spec The AdvNetConfigYaml object to populate.
   * @return true if decoding was successful, false otherwise.
   */
  static bool decode(const Node& node, holoscan::ops::AdvNetConfigYaml& input_spec) {
    if (!node.IsMap()) {
      GXF_LOG_ERROR("InputSpec: expected a map");
      return false;
    }

    // YAML is using exceptions, catch them
    try {
      input_spec.common_.version = node["version"].as<int32_t>();
      input_spec.common_.master_core_ = node["master_core"].as<int32_t>();
      try {
        input_spec.common_.manager_type = holoscan::ops::manager_type_from_string(
            node["manager"].as<std::string>(holoscan::ops::ANO_MGR_STR__DEFAULT));
      } catch (const std::exception& e) {
        input_spec.common_.manager_type = holoscan::ops::AnoMgrType::DEFAULT;
      }

      try {
        input_spec.debug_ = node["debug"].as<bool>(false);
      } catch (const std::exception& e) { input_spec.debug_ = false; }

      try {
        input_spec.log_level_ =
            holoscan::ops::AnoLogLevel::from_string(node["log_level"].as<std::string>(
                holoscan::ops::AnoLogLevel::to_string(holoscan::ops::AnoLogLevel::WARN)));
      } catch (const std::exception& e) {
        input_spec.log_level_ = holoscan::ops::AnoLogLevel::WARN;
      }

      try {
        const auto& mrs = node["memory_regions"];
        for (const auto& mr : mrs) {
          holoscan::ops::MemoryRegion tmr;
          if (!parse_memory_region_config(mr, tmr)) {
            HOLOSCAN_LOG_ERROR("Failed to parse memory region config");
            return false;
          }
          if (input_spec.mrs_.find(tmr.name_) != input_spec.mrs_.end()) {
            HOLOSCAN_LOG_CRITICAL("Duplicate memory region names: {}", tmr.name_);
            return false;
          }
          input_spec.mrs_[tmr.name_] = tmr;
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Must define at least one memory type");
        return false;
      }

      try {
        const auto& intfs = node["interfaces"];
        uint16_t port = 0;
        for (const auto& intf : intfs) {
          holoscan::ops::AdvNetConfigInterface ifcfg;

          ifcfg.name_ = intf["name"].as<std::string>();
          ifcfg.address_ = intf["address"].as<std::string>();
          try {
            ifcfg.flow_isolation_ = intf["flow_isolation"].as<bool>();
          } catch (const std::exception& e) { ifcfg.flow_isolation_ = false; }

          ifcfg.port_id_ = port++;

          const auto& rx = intf["rx"];
          for (const auto& rx_item : rx) {
            holoscan::ops::AdvNetRxConfig rx_cfg;

            for (const auto& q_item : rx_item["queues"]) {
              holoscan::ops::RxQueueConfig q;
              if (!parse_rx_queue_config(q_item, input_spec.common_.manager_type, q)) {
                HOLOSCAN_LOG_ERROR("Failed to parse RxQueueConfig");
                return false;
              }
              rx_cfg.queues_.emplace_back(std::move(q));
            }

            for (const auto& flow_item : rx_item["flows"]) {
              holoscan::ops::FlowConfig flow;
              if (!parse_flow_config(flow_item, flow)) {
                HOLOSCAN_LOG_ERROR("Failed to parse FlowConfig");
                return false;
              }
              rx_cfg.flows_.emplace_back(std::move(flow));
            }

            ifcfg.rx_ = rx_cfg;
          }

          const auto& tx = intf["tx"];
          for (const auto& tx_item : tx) {
            holoscan::ops::AdvNetTxConfig tx_cfg;

            try {
              tx_cfg.accurate_send_ = tx_item["accurate_send"].as<bool>();
            } catch (const std::exception& e) { tx_cfg.accurate_send_ = false; }

            for (const auto& q_item : tx_item["queues"]) {
              holoscan::ops::TxQueueConfig q;
              if (!parse_tx_queue_config(q_item, input_spec.common_.manager_type, q)) {
                HOLOSCAN_LOG_ERROR("Failed to parse TxQueueConfig");
                return false;
              }
              tx_cfg.queues_.emplace_back(std::move(q));
            }

            ifcfg.tx_ = tx_cfg;
          }

          input_spec.ifs_.push_back(ifcfg);
        }
      } catch (const std::exception& e) {
        GXF_LOG_ERROR(e.what());
        return false;
      }

      HOLOSCAN_LOG_INFO("Finished reading advanced network operator config");

      return true;
    } catch (const std::exception& e) {
      GXF_LOG_ERROR(e.what());
      return false;
    }
  }
};

namespace holoscan::ops {

  std::unordered_set<std::string> 
  adv_net_get_port_names(const Config& conf, const std::string& dir) {
    std::unordered_set<std::string> output_ports;
    std::string default_output_name;

    if (dir == "rx") {
      default_output_name = "bench_rx_out";
    } else if (dir == "tx") {
      default_output_name = "bench_tx_out";
    } else {
      return output_ports;
    }

    try {
      auto& yaml_nodes = conf.yaml_nodes();
      for (const YAML::Node& node : yaml_nodes) {
        const auto& intfs = node["advanced_network"]["cfg"]["interfaces"];
        for (const auto& intf : intfs) {
          const auto& intf_dir = intf[dir];
          for (const auto& dir_item : intf_dir) {
            for (const auto& q_item : dir_item["queues"]) {
              auto out_port_name = q_item["output_port"].as<std::string>(default_output_name);
              output_ports.insert(out_port_name);
            }
          }
        }
      }
    } catch (const std::exception& e) { GXF_LOG_ERROR(e.what()); }
    return output_ports;
  }

}