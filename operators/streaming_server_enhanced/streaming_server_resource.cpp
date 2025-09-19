/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:  // www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "streaming_server_resource.hpp"

#include <holoscan/core/component_spec.hpp>
#include <holoscan/holoscan.hpp>

#include <chrono>
#include <stdexcept>

namespace holoscan::ops {
StreamingServerResource::~StreamingServerResource() {
  if (streaming_server_ && is_running()) {
    try {
      stop();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error in destructor while stopping server: {}", e.what());
    }
  }
}

void StreamingServerResource::setup(ComponentSpec& spec) {
  spec.param(port_, "port", "Server Port", "Port for streaming server", uint16_t{48010});
  spec.param(is_multi_instance_, "is_multi_instance", "Multi Instance",
             "Allow multiple server instances", false);
  spec.param(server_name_, "server_name", "Server Name",
             "Server identifier", std::string("HoloscanStreamingServer"));
  spec.param(width_, "width", "Video Width", "Video frame width", 854U);
  spec.param(height_, "height", "Video Height", "Video frame height", 480U);
  spec.param(fps_, "fps", "Frame Rate", "Video frame rate", uint16_t{30});
  spec.param(enable_upstream_, "enable_upstream", "Enable Upstream",
             "Enable upstream (receiving) functionality", true);
  spec.param(enable_downstream_, "enable_downstream", "Enable Downstream",
             "Enable downstream (sending) functionality", true);
}

void StreamingServerResource::initialize() {
  Resource::initialize();

  // Build configuration from parameters
  config_.port = port_.has_value() ? port_.get() : 48010;
  config_.server_name = server_name_.has_value() ? server_name_.get() : "HoloscanStreamingServer";
  config_.width = width_.has_value() ? width_.get() : 854;
  config_.height = height_.has_value() ? height_.get() : 480;
  config_.fps = fps_.has_value() ? fps_.get() : 30;
  config_.enable_upstream = enable_upstream_.has_value() ? enable_upstream_.get() : true;
  config_.enable_downstream = enable_downstream_.has_value() ? enable_downstream_.get() : true;
  config_.multi_instance = is_multi_instance_.has_value() ? is_multi_instance_.get() : false;

  try {
    StreamingServer::Config server_config = to_streaming_server_config(config_);
    streaming_server_ = std::make_unique<StreamingServer>(server_config);

    // Only set event callback - this is the only callback that exists in the library
    streaming_server_->setEventCallback([this](const StreamingServer::Event& event) {
      handle_streaming_server_event(event);
    });

    is_initialized_ = true;
    start_time_ticks_ = std::chrono::steady_clock::now().time_since_epoch().count();

    HOLOSCAN_LOG_INFO(
        "StreamingServerResource initialized: {}:{}",
        config_.server_name,
        config_.port);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to initialize StreamingServerResource: {}", e.what());
    throw;
  }
}

void StreamingServerResource::start() {
  if (!is_initialized_ || !streaming_server_) {
    throw std::runtime_error("StreamingServerResource not initialized");
  }

  if (is_running()) {
    HOLOSCAN_LOG_WARN("StreamingServerResource already running");
    return;
  }

  try {
    HOLOSCAN_LOG_INFO("Starting StreamingServerResource on port {}", config_.port);

    // Start the StreamingServer (it handles all the Holoscan Streaming Stack complexity internally)
    streaming_server_->start();
    is_running_ = true;

    HOLOSCAN_LOG_INFO("✅ StreamingServerResource started successfully");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to start StreamingServerResource: {}", e.what());
    throw;
  }
}

void StreamingServerResource::stop() {
  if (!is_running()) {
    return;
  }

  try {
    HOLOSCAN_LOG_INFO("Stopping StreamingServerResource");

    if (streaming_server_) {
      streaming_server_->stop();
    }

    HOLOSCAN_LOG_INFO("StreamingServerResource stopped");
    is_running_ = false;  // Only set after successful stop
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error stopping StreamingServerResource: {}", e.what());
    // Don't re-throw to maintain compatibility with destructor and update_config()
    // The is_running() method checks streaming_server_->isRunning() for actual state
  }
}

bool StreamingServerResource::is_running() const {
  return is_running_ && streaming_server_ && streaming_server_->isRunning();
}

bool StreamingServerResource::has_connected_clients() const {
  return streaming_server_ ? streaming_server_->hasConnectedClients() : false;
}

void StreamingServerResource::send_frame(const Frame& frame) {
  if (!is_running() || !streaming_server_) {
    return;
  }

  try {
    if (config_.enable_downstream) {
      VideoFrame server_frame = convert_to_streaming_server_frame(frame);
      streaming_server_->sendFrame(server_frame);
      frames_sent_++;
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error sending frame: {}", e.what());
  }
}

Frame StreamingServerResource::receive_frame() {
  if (!is_running() || !streaming_server_) {
    return Frame{};
  }

  try {
    if (config_.enable_upstream) {
      VideoFrame server_frame = streaming_server_->receiveFrame();
      return convert_from_streaming_server_frame(server_frame);
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error receiving frame: {}", e.what());
  }

  return Frame{};
}

bool StreamingServerResource::try_receive_frame(Frame& frame) {
  if (!is_running() || !streaming_server_) {
    return false;
  }

  try {
    if (config_.enable_upstream) {
      VideoFrame server_frame;
      if (streaming_server_->tryReceiveFrame(server_frame)) {
        frame = convert_from_streaming_server_frame(server_frame);
        return true;
      }
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error trying to receive frame: {}", e.what());
  }

  return false;
}

StreamingServerResource::Config StreamingServerResource::get_config() const {
  return config_;
}

void StreamingServerResource::update_config(const Config& new_config) {
  // Note: Some config changes may require restart
  bool needs_restart = (config_.port != new_config.port) ||
                      (config_.width != new_config.width) ||
                      (config_.height != new_config.height) ||
                      (config_.multi_instance != new_config.multi_instance);

  if (needs_restart && is_running()) {
    HOLOSCAN_LOG_WARN("Configuration change requires server restart");
    stop();
    config_ = new_config;
    start();
  } else {
    config_ = new_config;
  }
}

StreamingServer::Config StreamingServerResource::get_streaming_server_config() const {
  return streaming_server_ ? streaming_server_->getConfig() : StreamingServer::Config{};
}

void StreamingServerResource::set_event_callback(EventCallback callback) {
  if (streaming_server_) {
    streaming_server_->setEventCallback(callback);
  }
}

bool StreamingServerResource::is_upstream_connected() const {
  return upstream_connected_;
}

bool StreamingServerResource::is_downstream_connected() const {
  return downstream_connected_;
}

size_t StreamingServerResource::get_connected_client_count() const {
  return connected_client_count_;
}

uint64_t StreamingServerResource::get_frames_received() const {
  return frames_received_;
}

uint64_t StreamingServerResource::get_frames_sent() const {
  return frames_sent_;
}

double StreamingServerResource::get_upstream_fps() const {
  if (!is_running() || frames_received_ == 0) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto start_time = std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(start_time_ticks_.load()));
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
  return elapsed.count() > 0 ? static_cast<double>(frames_received_) / elapsed.count() : 0.0;
}

double StreamingServerResource::get_downstream_fps() const {
  if (!is_running() || frames_sent_ == 0) {
    return 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  auto start_time = std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(start_time_ticks_.load()));
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
  return elapsed.count() > 0 ? static_cast<double>(frames_sent_) / elapsed.count() : 0.0;
}

void StreamingServerResource::handle_streaming_server_event(const StreamingServer::Event& event) {
  // Update internal state based on StreamingServer events
  switch (event.type) {
    case StreamingServer::EventType::ClientConnected:
      connected_client_count_++;
      break;
    case StreamingServer::EventType::ClientDisconnected:
      if (connected_client_count_ > 0) connected_client_count_--;
      break;
    case StreamingServer::EventType::UpstreamConnected:
      upstream_connected_ = true;
      break;
    case StreamingServer::EventType::UpstreamDisconnected:
      upstream_connected_ = false;
      break;
    case StreamingServer::EventType::DownstreamConnected:
      downstream_connected_ = true;
      break;
    case StreamingServer::EventType::DownstreamDisconnected:
      downstream_connected_ = false;
      break;
    case StreamingServer::EventType::FrameReceived:
      frames_received_++;
      break;
    case StreamingServer::EventType::FrameSent:
      frames_sent_++;
      break;
    default:
      break;
  }

  // Log events for debugging
  HOLOSCAN_LOG_DEBUG("StreamingServerResource event: {} - {}",
                    static_cast<int>(event.type), event.message);
}

StreamingServer::Config StreamingServerResource::to_streaming_server_config(const Config& config) const {
  StreamingServer::Config server_config;
  server_config.port = config.port;
  server_config.isMultiInstance = config.multi_instance;
  server_config.serverName = config.server_name;
  server_config.width = config.width;
  server_config.height = config.height;
  server_config.fps = config.fps;
  return server_config;
}

StreamingServerResource::Config StreamingServerResource::from_streaming_server_config(const StreamingServer::Config& server_config) const {
  Config config;
  config.port = server_config.port;
  config.multi_instance = server_config.isMultiInstance;
  config.server_name = server_config.serverName;
  config.width = server_config.width;
  config.height = server_config.height;
  config.fps = server_config.fps;
  // Note: enable_upstream and enable_downstream are resource-specific, not part of StreamingServer::Config
  config.enable_upstream = config_.enable_upstream;
  config.enable_downstream = config_.enable_downstream;
  return config;
}

VideoFrame StreamingServerResource::convert_to_streaming_server_frame(const Frame& ops_frame) const {
  // Create VideoFrame with dimensions and data
  VideoFrame server_frame(ops_frame.getWidth(), ops_frame.getHeight(),
                          ops_frame.getData(), ops_frame.getDataSize(),
                          ops_frame.getTimestamp());

  // Convert format - VideoFrame and ops_frame use the same PixelFormat enum
  server_frame.setFormat(ops_frame.getFormat());

  return server_frame;
}

Frame StreamingServerResource::convert_from_streaming_server_frame(const VideoFrame& server_frame) const {
  // Create Frame with dimensions and data
  Frame ops_frame(server_frame.getWidth(), server_frame.getHeight(),
                  server_frame.getData(), server_frame.getDataSize(),
                  server_frame.getTimestamp());

  // Convert format - VideoFrame and ops_frame use the same PixelFormat enum
  ops_frame.setFormat(server_frame.getFormat());

  return ops_frame;
}
}  // namespace holoscan::ops
