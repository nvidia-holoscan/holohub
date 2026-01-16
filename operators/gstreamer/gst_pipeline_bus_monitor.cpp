/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gst_pipeline_bus_monitor.hpp"

#include <holoscan/logger/logger.hpp>
#include "gst/bus.hpp"

namespace holoscan::gst {

PipelineBusMonitor::PipelineBusMonitor(Pipeline pipeline)
    : pipeline_(std::move(pipeline)), completion_future_(completion_promise_.get_future()) {}

PipelineBusMonitor::~PipelineBusMonitor() {
  stop();
}

void PipelineBusMonitor::start() {
  if (started_) {
    HOLOSCAN_LOG_WARN("Pipeline bus monitor already started");
    return;
  }

  stop_flag_.store(false);

  // Create a new promise/future for this monitoring session
  completion_promise_ = std::promise<void>();
  completion_future_ = completion_promise_.get_future();

  monitor_thread_ = std::async(std::launch::async, [this]() { monitor_loop(); });
  started_ = true;

  HOLOSCAN_LOG_DEBUG("Pipeline bus monitor started");
}

void PipelineBusMonitor::stop() {
  if (!started_) {
    return;
  }

  HOLOSCAN_LOG_DEBUG("Stopping pipeline bus monitor");
  stop_flag_.store(true);

  if (monitor_thread_.valid()) {
    monitor_thread_.wait();
  }

  started_ = false;
  HOLOSCAN_LOG_DEBUG("Pipeline bus monitor stopped");
}

bool PipelineBusMonitor::is_running() const {
  return started_ && !stop_flag_.load();
}

std::shared_future<void> PipelineBusMonitor::get_completion_future() {
  return completion_future_;
}

Pipeline& PipelineBusMonitor::get_pipeline() {
  return pipeline_;
}

void PipelineBusMonitor::on_error(const Error& error, const std::string& debug_info) {
  HOLOSCAN_LOG_ERROR("GStreamer error: {}", error->message);
  if (!debug_info.empty()) {
    HOLOSCAN_LOG_DEBUG("Debug info: {}", debug_info);
  }
}

void PipelineBusMonitor::on_eos() {
  HOLOSCAN_LOG_INFO("End of stream reached");
}

void PipelineBusMonitor::on_state_changed(GstState old_state, GstState new_state,
                                          GstState pending_state) {
  // Default implementation: check for unexpected NULL state transition
  if (new_state == GST_STATE_NULL && old_state != GST_STATE_NULL) {
    HOLOSCAN_LOG_INFO("Pipeline transitioned to NULL state");
  }
}

void PipelineBusMonitor::monitor_loop() {
  Bus bus = pipeline_.get_bus();

  while (!stop_flag_.load()) {
    Message msg = bus.timed_pop_filtered(
        100 * GST_MSECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS |
                                    GST_MESSAGE_STATE_CHANGED));

    if (msg) {
      switch (GST_MESSAGE_TYPE(msg.get())) {
        case GST_MESSAGE_ERROR: {
          std::string debug_info;
          auto error = msg.parse_error(debug_info);
          on_error(error, debug_info);

          // Error terminates monitoring
          completion_promise_.set_value();
          return;
        }

        case GST_MESSAGE_EOS:
          on_eos();

          // EOS terminates monitoring
          completion_promise_.set_value();
          return;

        case GST_MESSAGE_STATE_CHANGED: {
          // Only handle state changes from the pipeline itself (not individual elements)
          if (GST_MESSAGE_SRC(msg.get()) == GST_OBJECT(pipeline_.get())) {
            GstState old_state, new_state, pending_state;
            msg.parse_state_changed(&old_state, &new_state, &pending_state);

            on_state_changed(old_state, new_state, pending_state);

            // If pipeline transitions to NULL, stop monitoring
            if (new_state == GST_STATE_NULL && old_state != GST_STATE_NULL) {
              completion_promise_.set_value();
              return;
            }
          }
          break;
        }

        default:
          break;
      }
    }
  }

  // If we exit due to stop flag, still fulfill the promise
  completion_promise_.set_value();
}

}  // namespace holoscan::gst

