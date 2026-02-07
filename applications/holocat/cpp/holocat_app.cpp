/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

/**
 * @file holocat_app.cpp
 * @brief HoloCat Application Implementation
 *
 * Implementation of the main HoloCat application class that manages
 * EtherCAT integration with the Holoscan streaming framework.
 */

// System includes
#include <chrono>
#include <filesystem>

// Local includes
#include "hc_data_rx_op.hpp"
#include "hc_data_tx_op.hpp"
#include "holocat_app.hpp"
#include "holocat_op.hpp"

using namespace std::chrono_literals;

namespace holocat {

void HolocatApp::compose() {
  HOLOSCAN_LOG_INFO("HoloCat EtherCAT Application Starting...");

  // Extract and validate configuration parameters
  HolocatConfig config = extract_config();

  HOLOSCAN_LOG_INFO("EtherCAT adapter: {}", config.adapter_name);
  HOLOSCAN_LOG_INFO("Cycle time: {} μs", config.cycle_time_us);

  // Create and configure the HoloCat operator
  auto ecat_bus_periodic_cond = make_condition<holoscan::PeriodicCondition>(
      "ethercat_cycle_period", config.cycle_time_us * 1us);
  std::shared_ptr<HolocatOp> holocat_op =
      make_operator<HolocatOp>("holocat_op", ecat_bus_periodic_cond);
  holocat_op->set_config(config);
  add_operator(holocat_op);

  // Create and configure the HcDataTxOp operator
  auto counter_update_periodic_cond =
      make_condition<holoscan::PeriodicCondition>("ethercat_data_tx_period", 100ms);
  std::shared_ptr<HcDataTxOp> data_tx_op;
  data_tx_op = make_operator<HcDataTxOp>("data_tx_op", counter_update_periodic_cond);
  add_operator(data_tx_op);

  // create and configure the HcDataRxOp operator
  std::shared_ptr<HcDataRxOp> data_rx_op;
  data_rx_op = make_operator<HcDataRxOp>("data_rx_op");
  add_operator(data_rx_op);

  add_flow(data_tx_op, holocat_op, {{"count_out", "count_in"}});
  add_flow(holocat_op, data_rx_op, {{"count_out", "count_in"}});
}

/*
 * Extract configuration from the configuration file
 * @return The configuration object
 */
HolocatConfig HolocatApp::extract_config() {
  HolocatConfig config = {};

  // Helper lambda for safe configuration extraction
  auto try_extract = [this](const std::string& key, auto& target) {
    try {
      target = from_config(key).as<std::decay_t<decltype(target)>>();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_WARN("Configuration key '{}': {}", key, e.what());
    }
  };

  // EtherCAT network configuration
  try_extract("holocat.adapter_name", config.adapter_name);
  try_extract("holocat.eni_file", config.eni_file);

  // Timing configuration
  try_extract("holocat.cycle_time_us", config.cycle_time_us);

  // Real-time configuration
  try_extract("holocat.rt_priority", config.rt_priority);
  try_extract("holocat.job_thread_priority", config.job_thread_priority);
  try_extract("holocat.enable_rt", config.enable_rt);
  try_extract("holocat.job_thread_stack_size", config.job_thread_stack_size);

  // Process data configuration
  try_extract("holocat.dio_out_offset", config.dio_out_offset);
  try_extract("holocat.dio_in_offset", config.dio_in_offset);

  // Communication configuration
  try_extract("holocat.max_acyc_frames", config.max_acyc_frames);

  // Logging configuration
  try_extract("holoscan.logging.level", config.log_level);

  // Validate and finalize configuration
  if (!validate_config(config)) {
    HOLOSCAN_LOG_ERROR("Configuration validation failed: {}", config.error_message);
    throw std::runtime_error("Invalid configuration: " + config.error_message);
  }

  return config;
}

/*
 * Validate the configuration
 * @param config The configuration object
 * @return True if the configuration is valid, false otherwise
 */
bool HolocatApp::validate_config(HolocatConfig& config) {
  // Validate adapter name
  if (config.adapter_name.empty()) {
    config.error_message = "EtherCAT adapter name cannot be empty";
    return false;
  }

  // Resolve ENI file path relative to config file directory if it's a relative path
  std::filesystem::path eni_path(config.eni_file);
  // Check if eni_path is empty
  if (eni_path.empty()) {
    config.error_message = "ENI file path cannot be empty";
    return false;
  }
  if (eni_path.is_relative()) {
    // Get the directory containing the config file
    std::filesystem::path config_dir =
        std::filesystem::path(this->config().config_file()).parent_path();
    // Resolve the ENI file path relative to the config directory
    eni_path = config_dir / eni_path;
    // Update config with the resolved absolute path
    config.eni_file = eni_path.string();
    HOLOSCAN_LOG_INFO("Resolved relative ENI path to: {}", config.eni_file);
  }

  // Validate ENI file exists
  std::error_code ec;
  if (!std::filesystem::exists(config.eni_file, ec)) {
    if (ec) {
      config.error_message =
          "Unable to access ENI file: " + config.eni_file + " (" + ec.message() + ")";
    } else {
      config.error_message = "ENI configuration file not found: " + config.eni_file;
    }
    return false;
  }

  // Validate cycle time range
  constexpr uint64_t MIN_CYCLE_TIME_US = 100;
  constexpr uint64_t MAX_CYCLE_TIME_US = 100000;
  if (config.cycle_time_us < MIN_CYCLE_TIME_US || config.cycle_time_us > MAX_CYCLE_TIME_US) {
    config.error_message = "Invalid cycle time: " + std::to_string(config.cycle_time_us) +
                           " μs (valid range: " + std::to_string(MIN_CYCLE_TIME_US) + "-" +
                           std::to_string(MAX_CYCLE_TIME_US) + " μs)";
    return false;
  }

  // Validate and correct RT priorities
  constexpr uint32_t MIN_PRIORITY = 1;
  constexpr uint32_t MAX_PRIORITY = 99;

  if (config.rt_priority < MIN_PRIORITY || config.rt_priority > MAX_PRIORITY) {
    config.error_message = "Invalid RT priority: " + std::to_string(config.rt_priority) +
                           " (valid range: " + std::to_string(MIN_PRIORITY) + "-" +
                           std::to_string(MAX_PRIORITY) + ")";
    return false;
  }

  if (config.job_thread_priority < MIN_PRIORITY || config.job_thread_priority > MAX_PRIORITY) {
    config.error_message =
        "Invalid job thread priority: " + std::to_string(config.job_thread_priority) +
        " (valid range: " + std::to_string(MIN_PRIORITY) + "-" + std::to_string(MAX_PRIORITY) + ")";
    return false;
  }

  constexpr uint32_t MIN_ACYC_FRAMES = 1;
  constexpr uint32_t MAX_ACYC_FRAMES = 32;
  if (config.max_acyc_frames < MIN_ACYC_FRAMES || config.max_acyc_frames > MAX_ACYC_FRAMES) {
    config.error_message = "Invalid max acyclic frames: " + std::to_string(config.max_acyc_frames) +
                           " (valid range: " + std::to_string(MIN_ACYC_FRAMES) + "-" +
                           std::to_string(MAX_ACYC_FRAMES) + ")";
    return false;
  }

  constexpr uint32_t MIN_STACK_SIZE = 0x8000;
  constexpr uint32_t MAX_STACK_SIZE = 0x10000;
  if (config.job_thread_stack_size < MIN_STACK_SIZE ||
      config.job_thread_stack_size > MAX_STACK_SIZE) {
    config.error_message =
        "Invalid job thread stack size: " + std::to_string(config.job_thread_stack_size) +
        " (valid range: 0x8000-0x10000)";
    return false;
  }

  return true;
}

}  // namespace holocat
