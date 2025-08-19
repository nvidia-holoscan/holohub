/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <getopt.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

// Simple data structure for ping messages
struct PingMessage {
  int index;
  uint64_t timestamp;
};

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<PingMessage>("out"); }

  void processing_time_ms(unsigned int processing_time_ms) {
    processing_time_ms_ = processing_time_ms;
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    PingMessage msg;
    msg.index = index_;
    msg.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();

    // Busy-loop for 5ms before emitting
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - start_time <
           std::chrono::milliseconds(processing_time_ms_)) {
      // Busy wait
    }

    HOLOSCAN_LOG_INFO("Ping TX: Generated index {}", index_);
    op_output.emit(msg, "out");

    index_++;
  }

 private:
  unsigned int processing_time_ms_ = 0;
  int index_ = 0;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<PingMessage>("in1");
    spec.input<PingMessage>("in2");
  }

  void max_index(int max_index) { max_index_ = max_index; }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    static int in1_index = -1;
    static uint64_t in1_timestamp = 0;
    static int in2_index = -1;
    static uint64_t in2_timestamp = 0;
    static std::ofstream tx1_file("tx1.csv");
    static std::ofstream in1_periods("rx_in1_periods.csv");
    static std::ofstream tx2_file("tx2.csv");
    static std::ofstream in2_periods("rx_in2_periods.csv");
    static bool files_initialized = false;

    // Initialize CSV files with headers if not done yet
    if (!files_initialized) {
      tx1_file << "index,latency_ms" << std::endl;
      tx2_file << "index,latency_ms" << std::endl;
      in1_periods << "index,period_ms" << std::endl;
      in2_periods << "index,period_ms" << std::endl;
      files_initialized = true;
    }

    auto maybe_msg = op_input.receive<PingMessage>("in1");
    if (!maybe_msg) {
      HOLOSCAN_LOG_INFO("Operator '{}' did not receive a valid message.", this->name());
      return;
    } else {
      auto msg = maybe_msg.value();

      if (msg.index != in1_index) {
        in1_index = msg.index;

        uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::high_resolution_clock::now().time_since_epoch())
                                    .count();
        if (in1_timestamp != 0) {
          double observed_period = static_cast<double>(current_time - in1_timestamp) / 1000.0;
          in1_periods << msg.index << "," << std::fixed << std::setprecision(2) << observed_period
                      << std::endl;
          in1_periods.flush();
        }
        in1_timestamp = current_time;

        uint64_t latency_us = current_time - msg.timestamp;
        double latency_ms = static_cast<double>(latency_us) / 1000.0;

        // Log to tx1.csv
        tx1_file << msg.index << "," << std::fixed << std::setprecision(2) << latency_ms
                 << std::endl;
        tx1_file.flush();  // Ensure data is written immediately

        HOLOSCAN_LOG_INFO(
            "Ping RX1: Received index {} (latency: {:.3f} ms)", msg.index, latency_ms);
      }
    }

    maybe_msg = op_input.receive<PingMessage>("in2");
    if (!maybe_msg) {
      HOLOSCAN_LOG_INFO("Operator '{}' did not receive a valid message.", this->name());
      return;
    } else {
      auto msg = maybe_msg.value();

      if (msg.index != in2_index) {
        in2_index = msg.index;

        uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::high_resolution_clock::now().time_since_epoch())
                                    .count();
        if (in2_timestamp != 0) {
          double observed_period = static_cast<double>(current_time - in2_timestamp) / 1000.0;
          in2_periods << msg.index << "," << std::fixed << std::setprecision(2) << observed_period
                      << std::endl;
          in2_periods.flush();
        }
        in2_timestamp = current_time;
        uint64_t latency_us = current_time - msg.timestamp;
        double latency_ms = static_cast<double>(latency_us) / 1000.0;

        // Log to tx2.csv
        tx2_file << msg.index << "," << std::fixed << std::setprecision(2) << latency_ms
                 << std::endl;
        tx2_file.flush();  // Ensure data is written immediately

        HOLOSCAN_LOG_INFO(
            "Ping RX2: Received index {} (latency: {:.3f} ms)", msg.index, latency_ms);
      }
    }

    // Busy-loop for 10ms after receiving
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - start_time < std::chrono::milliseconds(1)) {
      // Busy wait
    }

    // Check if both ports have received the max index and disable the condition
    if (in1_index == max_index_ - 1 && in2_index == max_index_ - 1) {
      HOLOSCAN_LOG_INFO("Both ports received max index, disabling boolean condition");
      // Get the boolean condition from the operator's condition and cast to BooleanCondition
      auto condition = this->condition<holoscan::BooleanCondition>("is_done");
      if (condition) {
        condition->disable_tick();
      } else {
        HOLOSCAN_LOG_ERROR("Failed to get BooleanCondition 'is_done'");
      }
    }
  }

 private:
  int max_index_ = 0;
};

}  // namespace holoscan::ops

class PingAsyncBufferApp : public holoscan::Application {
 public:
  PingAsyncBufferApp(int num_messages = 100, bool enable_async_buffer = false,
                     int tx1_period_ms = 20, int tx2_period_ms = 20)
      : num_messages_(num_messages),
        enable_async_buffer_(enable_async_buffer),
        tx1_period_ms_(tx1_period_ms),
        tx2_period_ms_(tx2_period_ms) {}

  void compose() override {
    using namespace holoscan;

    auto tx1 =
        make_operator<ops::PingTxOp>("ping_tx1", make_condition<CountCondition>(num_messages_));
    auto tx2 =
        make_operator<ops::PingTxOp>("ping_tx2", make_condition<CountCondition>(num_messages_));
    auto rx =
        make_operator<ops::PingRxOp>("ping_rx", make_condition<BooleanCondition>("is_done", true));

    rx->max_index(num_messages_);

    tx1->processing_time_ms(5);
    tx2->processing_time_ms(10);

    // Connect them with optional async buffer
    if (enable_async_buffer_) {
      add_flow(tx1, rx, {{"out", "in1"}}, IOSpec::ConnectorType::kAsyncBuffer);
      add_flow(tx2, rx, {{"out", "in2"}}, IOSpec::ConnectorType::kAsyncBuffer);
    } else {
      add_flow(tx1, rx, {{"out", "in1"}});
      add_flow(tx2, rx, {{"out", "in2"}});
    }

    // Create thread pool for deadline scheduling
    auto pool = make_thread_pool("deadline_pool", 0);

    // Add operators to thread pool with deadline scheduling
    // all our supported platforms have at least 4 CPU cores, so we can use 1-3
    // for three operators.
    pool->add_realtime(tx1,
                       holoscan::SchedulingPolicy::kDeadline,
                       true,
                       {1},
                       0,
                       10000000,
                       tx1_period_ms_ * 1000000,
                       tx1_period_ms_ * 1000000);
    pool->add_realtime(tx2,
                       holoscan::SchedulingPolicy::kDeadline,
                       true,
                       {2},
                       0,
                       15000000,
                       tx2_period_ms_ * 1000000,
                       tx2_period_ms_ * 1000000);
    // PingRxOp: 8ms budget every 10ms period
    pool->add_realtime(
        rx, holoscan::SchedulingPolicy::kDeadline, true, {3}, 0, 8000000, 10000000, 10000000);
  }

 private:
  int num_messages_;
  bool enable_async_buffer_;
  int tx1_period_ms_;
  int tx2_period_ms_;
};

void print_help(const char* program_name) {
  std::cout
      << "Ping Async Buffer with Deadline" << std::endl
      << std::endl
      << "Usage: " << program_name << " [options]" << std::endl
      << std::endl
      << "Options:" << std::endl
      << "  -h, --help                    Display this information" << std::endl
      << "  -m <COUNT>, --messages <COUNT> Number of messages to send (default: 100)" << std::endl
      << "  -a, --async-buffer            Enable async buffer connector" << std::endl
      << "  -x <MS>, --tx1-period <MS>    Set TX1 period in milliseconds (default: 20, min: 10)"
      << std::endl
      << "  -y <MS>, --tx2-period <MS>    Set TX2 period in milliseconds (default: 20, min: 15)"
      << std::endl
      << std::endl;
}

int main(int argc, char** argv) {
  // Check for message count and other options
  int num_messages = 100;  // Default value
  bool enable_async_buffer = false;
  int tx1_period_ms = 20;
  int tx2_period_ms = 20;

  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"messages", required_argument, 0, 'm'},
                                  {"async-buffer", no_argument, 0, 'a'},
                                  {"tx1-period", required_argument, 0, 'x'},
                                  {"tx2-period", required_argument, 0, 'y'},
                                  {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hm:a:x:y:", long_options, &option_index);

    if (c == -1) {
      break;
    }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        print_help(argv[0]);
        return 0;

      case 'm':
        num_messages = std::stoi(argument);
        break;

      case 'a':
        enable_async_buffer = true;
        break;

      case 'x':
        tx1_period_ms = std::stoi(argument);
        if (tx1_period_ms < 10) {
          std::cerr << "Error: TX1 period must be at least 10ms (got " << tx1_period_ms << "ms)"
                    << std::endl;
          return 1;
        }
        break;

      case 'y':
        tx2_period_ms = std::stoi(argument);
        if (tx2_period_ms < 15) {
          std::cerr << "Error: TX2 period must be at least 15ms (got " << tx2_period_ms << "ms)"
                    << std::endl;
          return 1;
        }
        break;

      case '?':
      default:
        std::cerr << std::endl;
        print_help(argv[0]);
        return 1;
    }
  }

  auto app = holoscan::make_application<PingAsyncBufferApp>(
      num_messages, enable_async_buffer, tx1_period_ms, tx2_period_ms);

  // Set up event-based scheduler with 2 threads
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("event-scheduler"));

  if (enable_async_buffer) {
    HOLOSCAN_LOG_INFO("Async buffer connector enabled");
  }
  HOLOSCAN_LOG_INFO("Number of messages: {}", num_messages);

  app->run();

  return 0;
}
