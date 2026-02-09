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
 * @file test_integration_hardware.cpp
 * @brief Hardware Integration Test for HoloCat
 *
 * Simple test that verifies EtherCAT hardware communication with data loopback.
 * Test is automatically skipped if hardware is not available.
 *
 * Environment variables required:
 *   ECMASTER_ROOT - Path to EC-Master SDK
 *   HOLOCAT_TEST_ADAPTER - Network adapter (e.g., eth0)
 *   HOLOCAT_TEST_ENI - Path to ENI configuration file
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "hc_data_rx_op.hpp"
#include "hc_data_tx_op.hpp"
#include "holocat_config.hpp"
#include "holocat_op.hpp"
#include "holoscan/holoscan.hpp"

using namespace std::chrono_literals;

namespace holocat {
namespace hardware_test {

// Global variables for test data verification
std::atomic<int> transmitted_value_{42};
std::atomic<int> last_count_{0};

/**
 * @brief Check if hardware testing prerequisites are met
 */
bool CheckPrerequisites(std::string& error) {
  const char* ecmaster_root = getenv("ECMASTER_ROOT");
  if (!ecmaster_root || std::string(ecmaster_root).empty()) {
    error = "ECMASTER_ROOT not set";
    return false;
  }

  const char* adapter = getenv("HOLOCAT_TEST_ADAPTER");
  if (!adapter || std::string(adapter).empty()) {
    error = "HOLOCAT_TEST_ADAPTER not set";
    return false;
  }

  const char* eni_file = getenv("HOLOCAT_TEST_ENI");
  if (!eni_file || std::string(eni_file).empty()) {
    error = "HOLOCAT_TEST_ENI not set";
    return false;
  }

  return true;
}

/**
 * @brief Get hardware test configuration from environment
 */
HolocatConfig GetHardwareConfig() {
  if (getenv("HOLOCAT_TEST_ADAPTER") == nullptr || getenv("HOLOCAT_TEST_ENI") == nullptr) {
    throw std::runtime_error("HOLOCAT_TEST_ADAPTER or HOLOCAT_TEST_ENI not set in environment");
  }
  HolocatConfig config{};
  config.adapter_name = getenv("HOLOCAT_TEST_ADAPTER");
  config.eni_file = getenv("HOLOCAT_TEST_ENI");
  config.enable_rt = false;
  config.rt_priority = 39;
  config.job_thread_priority = 98;
  config.cycle_time_us = 1000;
  config.dio_out_offset = 80;
  config.dio_in_offset = 144;
  config.max_acyc_frames = 32;
  config.job_thread_stack_size = 0x8000;
  config.log_level = "info";
  return config;
}

// subclass of HcDataTxOp that emits one value (42) and then stops
class OneValueTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OneValueTxOp);
  OneValueTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<int>("count_out"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    op_output.emit<int>(transmitted_value_, "count_out");
  }
};

class LoopbackCheckRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(LoopbackCheckRxOp);
  LoopbackCheckRxOp() = default;
  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("count_in"); }
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    auto maybe_count = op_input.receive<int>("count_in");
    if (maybe_count) {
      last_count_.store(maybe_count.value());
    } else {
      HOLOSCAN_LOG_ERROR("LoopbackCheckRxOp: Failed to receive count from ECat bus");
    }
  }
};

/**
 * @brief Simple loopback test application
 */
class LoopbackApp : public holoscan::Application {
 public:
  explicit LoopbackApp(const HolocatConfig& config) : config_(config) {}

  void compose() override {
    using namespace holoscan;

    auto tx_op =
        make_operator<OneValueTxOp>("tx_op", make_condition<PeriodicCondition>("periodic", 100ms));

    auto holocat_op =
        make_operator<HolocatOp>("holocat_op",
                                 make_condition<PeriodicCondition>("ethercat_cycle_period", 10ms),
                                 make_condition<CountCondition>("count", 300));
    holocat_op->set_config(config_);

    auto rx_op = make_operator<LoopbackCheckRxOp>("rx_op");

    add_flow(tx_op, holocat_op, {{"count_out", "count_in"}});
    add_flow(holocat_op, rx_op, {{"count_out", "count_in"}});
  }

 private:
  HolocatConfig config_;
};

/**
 * @brief Hardware integration test fixture
 */
class HardwareTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string error;
    if (!CheckPrerequisites(error)) {
      GTEST_SKIP() << std::endl
                   << " >>> Hardware prerequisites not met: " << error << std::endl
                   << std::endl;
    }
  }
};

/**
 * @brief Test data loopback through EtherCAT hardware
 */
TEST_F(HardwareTest, DataLoopback) {
  auto config = GetHardwareConfig();
  auto app = std::make_shared<LoopbackApp>(config);

  std::cout << "\n=== Hardware Loopback Test ===\n";
  std::cout << "Adapter: " << config.adapter_name << "\n";
  HOLOSCAN_LOG_INFO("ENI: {}", config.eni_file);
  std::cout << "Cycle: " << config.cycle_time_us << " μs\n";
  std::cout << "Max cycles: 1000\n\n";

  std::atomic<bool> app_failed{false};
  std::string error_msg;

  std::thread app_thread([&]() {
    try {
      app->run();
    } catch (const std::exception& e) {
      error_msg = e.what();
      app_failed.store(true, std::memory_order_release);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  if (app_failed.load(std::memory_order_acquire)) {
    if (app_thread.joinable())
      app_thread.join();
    GTEST_SKIP() << "Hardware not available: " << error_msg;
  }

  if (app_thread.joinable()) {
    app_thread.join();
  }

  if (app_failed.load(std::memory_order_acquire)) {
    FAIL() << "App failed: " << error_msg;
  }

  EXPECT_EQ(last_count_.load(), transmitted_value_.load())
      << "Received value (" << last_count_.load() << ") does not match transmitted value ("
      << transmitted_value_.load() << ")";

  std::cout << "✓ Application ran successfully\n";
  std::cout << "✓ Hardware loopback test passed\n";
  std::cout << "================================\n\n";

  SUCCEED();
}

}  // namespace hardware_test
}  // namespace holocat
