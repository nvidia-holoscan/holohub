/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/core/resource.hpp>
#include <holoscan/core/parameter.hpp>

#include <memory>
#include <functional>
#include <string>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>

#include "StreamingServer.h"
#include "VideoFrame.h"

namespace holoscan::ops {

// Use VideoFrame as our Frame type for better API consistency
using Frame = VideoFrame;

// Provide a mapping between VideoFrame::PixelFormat and our local PixelFormat
namespace FrameFormat {
    enum Format {
        BGR = static_cast<int>(PixelFormat::BGR),
        BGRA = static_cast<int>(PixelFormat::BGRA),
        RGBA = static_cast<int>(PixelFormat::RGBA)
        // Note: If you need RGB (3-channel), use BGR format and handle channel ordering
        // in your code
    };
}

/**
 * @brief Holoscan resource for managing streaming server connections
 *
 * This resource acts as a centralized manager for streaming server operations,
 * handling both upstream (receiving) and downstream (sending) connections.
 * It provides a Holoscan-native interface to the StreamingServer class which
 * already implements the PIMPL pattern for Holoscan Streaming Stack abstraction.
 *
 * The resource can be shared between multiple operators (e.g., upstream and downstream)
 * to coordinate streaming operations on a single server instance.
 */
class StreamingServerResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(StreamingServerResource)

  // Type aliases for StreamingServer types with proper namespace
  using Event = StreamingServer::Event;
  using EventType = StreamingServer::EventType;
  using EventCallback = StreamingServer::EventCallback;
  using FrameReceivedCallback = StreamingServer::FrameReceivedCallback;

  // Configuration structure that mirrors StreamingServer::Config
  struct Config {
    uint16_t port = 48010;
    std::string server_name = "HoloscanStreamingServer";
    uint32_t width = 854;
    uint32_t height = 480;
    uint16_t fps = 30;
    bool enable_upstream = true;
    bool enable_downstream = true;
    bool multi_instance = false;
  };

  StreamingServerResource() = default;
  ~StreamingServerResource() override;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  // Core streaming server operations
  void start();
  void stop();
  bool is_running() const;
  bool has_connected_clients() const;

  // Frame operations - using the operator Frame type for consistency
  void send_frame(const Frame& frame);
  Frame receive_frame();
  bool try_receive_frame(Frame& frame);

  // Configuration management
  Config get_config() const;
  void update_config(const Config& config);
  StreamingServer::Config get_streaming_server_config() const;

  // Event and callback management
  void set_event_callback(EventCallback callback);  // For backward compatibility
  void add_event_listener(EventCallback callback);  // For multiple listeners
  // Note: setFrameReceivedCallback is not available in the current StreamingServer library

  // Status and metrics
  bool is_upstream_connected() const;
  bool is_downstream_connected() const;
  size_t get_connected_client_count() const;
  uint64_t get_frames_received() const;
  uint64_t get_frames_sent() const;
  double get_upstream_fps() const;
  double get_downstream_fps() const;

  // Direct access to underlying StreamingServer (if needed)
  StreamingServer* get_streaming_server() const { return streaming_server_.get(); }

 private:
  // Configuration parameters
  Parameter<uint16_t> port_;
  Parameter<bool> is_multi_instance_;
  Parameter<std::string> server_name_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint16_t> fps_;
  Parameter<bool> enable_upstream_;
  Parameter<bool> enable_downstream_;

  // StreamingServer instance (uses its own PIMPL internally)
  std::unique_ptr<StreamingServer> streaming_server_;

  // Resource configuration
  Config config_;

  // Internal state
  std::atomic<bool> is_initialized_{false};
  std::atomic<bool> is_running_{false};
  std::atomic<bool> upstream_connected_{false};
  std::atomic<bool> downstream_connected_{false};
  std::atomic<size_t> connected_client_count_{0};

  // Performance tracking
  std::atomic<uint64_t> frames_received_{0};
  std::atomic<uint64_t> frames_sent_{0};
  std::atomic<std::chrono::steady_clock::time_point::rep> start_time_ticks_{0};

  // Event listener management
  std::vector<EventCallback> event_listeners_;
  std::mutex event_listeners_mutex_;

  // Internal event handling
  void handle_streaming_server_event(const StreamingServer::Event& event);

  // Convert between Config types
  StreamingServer::Config to_streaming_server_config(const Config& config) const;
  Config from_streaming_server_config(const StreamingServer::Config& config) const;

  // Convert between Frame types
  VideoFrame convert_to_streaming_server_frame(const Frame& ops_frame) const;
  Frame convert_from_streaming_server_frame(const VideoFrame& server_frame) const;
};

}  // namespace holoscan::ops
