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

#ifndef HOLOSCAN_BENCHMARK
#define HOLOSCAN_BENCHMARK

#include <stdlib.h>
#include <string>
#include <unordered_set>

#include "holoscan/holoscan.hpp"

class BenchmarkedApplication : public holoscan::Application {
 public:
  inline void add_flow(const std::shared_ptr<holoscan::Operator>& upstream_op,
                       const std::shared_ptr<holoscan::Operator>& downstream_op) override {
    this->add_flow(upstream_op, downstream_op, {});
  }

  inline void add_flow(const std::shared_ptr<holoscan::Operator>& upstream_op,
                       const std::shared_ptr<holoscan::Operator>& downstream_op,
                       std::set<std::pair<std::string, std::string>> port_pairs) override {
    // Call the parent add_flow
    Fragment::add_flow(upstream_op, downstream_op, port_pairs);

    // Add a CountCondition to an operator so that the application stops after a certain number of
    // messages.
    if (conditioned_nodes_.find(upstream_op) == conditioned_nodes_.end()) {
      // Load the number of source messages from HOLOSCAN_NUM_SOURCE_MESSAGES
      const char* src_frame_str = std::getenv("HOLOSCAN_NUM_SOURCE_MESSAGES");
      if (src_frame_str) { num_source_messages_ = std::stoi(src_frame_str); }

      conditioned_nodes_.insert(upstream_op);
      upstream_op->add_arg(make_condition<holoscan::CountCondition>(num_source_messages_));
    }
  }

  inline void run() override {
    // Enable data flow tracking
    if (!data_flow_tracker()) track();
    tracker_ = data_flow_tracker();
    // Get the data flow tracking logging file name from the environment variable
    const char* flow_tracking_log_file = std::getenv("HOLOSCAN_FLOW_TRACKING_LOG_FILE");
    if (!flow_tracking_log_file) {
      tracker_->enable_logging();
    } else {
      tracker_->enable_logging(flow_tracking_log_file);
    }

    // Load scheduler parameters from environment variables
    const char* scheduler_str = std::getenv("HOLOSCAN_SCHEDULER");
    if (scheduler_str && std::string(scheduler_str) == "multithread") {
      holoscan::Fragment::scheduler(
          holoscan::Fragment::make_scheduler<holoscan::MultiThreadScheduler>(
              "multithread-scheduler"));
      auto scheduler = holoscan::Fragment::scheduler();

      const char* num_threads_str = std::getenv("HOLOSCAN_MULTITHREAD_WORKER_THREADS");
      if (num_threads_str)
        scheduler->add_arg(holoscan::Arg("worker_thread_number", std::stoi(num_threads_str)));

      // MultiThread Scheduler default values
      scheduler->add_arg(holoscan::Arg("stop_on_deadlock", true));
      scheduler->add_arg(holoscan::Arg("check_recession_period_ms", (double)0));
      scheduler->add_arg(holoscan::Arg("max_duration_ms", (int64_t)100000));
    } else {
      holoscan::Fragment::scheduler(
          holoscan::Fragment::make_scheduler<holoscan::GreedyScheduler>("greedy-scheduler"));
    }

    // Call the parent's class' run()
    holoscan::Application::run();
  }
  ~BenchmarkedApplication() { /*tracker_->print();*/
  }

 private:
  holoscan::DataFlowTracker* tracker_ = nullptr;
  std::unordered_set<std::shared_ptr<holoscan::Operator>> conditioned_nodes_;
  int num_source_messages_ = 100;
};

#endif /* HOLOSCAN_BENCHMARK */
