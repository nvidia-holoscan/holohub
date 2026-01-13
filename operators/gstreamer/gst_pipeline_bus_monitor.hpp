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

#ifndef GST_PIPELINE_BUS_MONITOR_HPP
#define GST_PIPELINE_BUS_MONITOR_HPP

#include <atomic>
#include <future>
#include <memory>

#include "gst/pipeline.hpp"
#include "gst/message.hpp"

namespace holoscan::gst {

/**
 * @brief Base class for monitoring GStreamer pipeline bus messages
 *
 * This class provides a thread-based monitoring system for GStreamer pipeline bus messages.
 * It polls the bus for messages and dispatches them to virtual handler functions that can
 * be overridden by derived classes to implement custom behavior.
 *
 * The monitor runs in a separate thread and can be started/stopped gracefully.
 * By default, it handles ERROR, EOS, and STATE_CHANGED messages, but can be extended
 * to handle additional message types.
 *
 * @note Thread Safety: The public API functions (start(), stop(), is_running()) are NOT
 * thread-safe and should not be called concurrently from multiple threads. They should
 * be called from a single controlling thread.
 *
 * Example usage:
 * @code
 * class MyMonitor : public PipelineBusMonitor {
 * protected:
 *   void on_error(const Error& error, const std::string& debug_info) override {
 *     // Custom error handling
 *     HOLOSCAN_LOG_ERROR("Custom: {}", error->message);
 *   }
 * };
 *
 * MyMonitor monitor(pipeline);
 * monitor.start();
 * // ... do work ...
 * monitor.stop();
 * @endcode
 */
class PipelineBusMonitor {
 public:
  /**
   * @brief Construct a pipeline bus monitor
   *
   * @param pipeline The GStreamer pipeline to monitor (copied to manage refcount)
   */
  explicit PipelineBusMonitor(Pipeline pipeline);

  /**
   * @brief Virtual destructor
   *
   * Ensures the monitoring thread is stopped before destruction.
   */
  virtual ~PipelineBusMonitor();

  // Delete copy constructor and assignment operator
  PipelineBusMonitor(const PipelineBusMonitor&) = delete;
  PipelineBusMonitor& operator=(const PipelineBusMonitor&) = delete;

  /**
   * @brief Start monitoring the pipeline bus in a background thread
   *
   * This spawns a new thread that continuously polls the bus for messages.
   * If monitoring is already running, this method does nothing.
   */
  void start();

  /**
   * @brief Stop monitoring the pipeline bus
   *
   * Signals the monitoring thread to stop and waits for it to complete.
   * This method is safe to call multiple times.
   */
  void stop();

  /**
   * @brief Check if the monitor is currently running
   *
   * @return true if the monitoring thread is active, false otherwise
   */
  bool is_running() const;

  /**
   * @brief Get the future for monitoring completion
   *
   * This can be used to wait for the monitor to finish (e.g., on EOS or error).
   *
   * @return Shared future that completes when monitoring stops
   */
  std::shared_future<void> get_completion_future();

 protected:
  /**
   * @brief Handler for GStreamer error messages
   *
   * Override this method to customize error handling behavior.
   * Default implementation logs the error and stops monitoring.
   *
   * @param error The GError object containing error details
   * @param debug_info Additional debug information (may be empty)
   */
  virtual void on_error(const Error& error, const std::string& debug_info);

  /**
   * @brief Handler for End-Of-Stream messages
   *
   * Override this method to customize EOS handling behavior.
   * Default implementation logs a message and stops monitoring.
   */
  virtual void on_eos();

  /**
   * @brief Handler for pipeline state change messages
   *
   * Override this method to customize state change handling behavior.
   * Default implementation checks for transition to NULL state and stops monitoring.
   *
   * @param old_state The previous pipeline state
   * @param new_state The new pipeline state
   * @param pending_state The pending pipeline state (if any)
   */
  virtual void on_state_changed(GstState old_state, GstState new_state, GstState pending_state);

  /**
   * @brief Get a reference to the monitored pipeline
   *
   * @return Reference to the pipeline being monitored
   */
  Pipeline& get_pipeline();

 private:
  /**
   * @brief Main monitoring loop (runs in separate thread)
   *
   * Continuously polls the bus for messages until stopped.
   */
  void monitor_loop();

  Pipeline pipeline_;  // Store pipeline object to manage GObject refcount
  std::atomic<bool> stop_flag_{false};
  std::promise<void> completion_promise_;
  std::shared_future<void> completion_future_;
  std::future<void> monitor_thread_;
  bool started_{false};
};

}  // namespace holoscan::gst

#endif  // GST_PIPELINE_BUS_MONITOR_HPP

