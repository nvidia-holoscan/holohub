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

#include <memory>
#include <functional>
#include <vector>
#include <cstdint>
#include <string>
#include "VideoFrame.h"  // Use VideoFrame throughout

/**
 * @brief StreamingServer class that unifies all StreamSDK functionality
 * 
 * This class provides a clean API for bidirectional video streaming,
 * hiding all Stream SDK implementation details through the PIMPL pattern.
 */
class StreamingServer
{
public:
    /**
     * @brief Configuration structure for server settings
     */
    struct Config {
        // Server settings
        uint16_t port = 48010;              // Server port
        bool isMultiInstance = false;       // Allow multiple server instances
        std::string serverName = "StreamingServer"; // Name identifier for the server
        
        // Video settings
        uint32_t width = 854;               // Video width
        uint32_t height = 480;              // Video height
        uint16_t fps = 30;                  // Video frame rate
    };
    
    /**
     * @brief Event types for streaming events
     */
    enum class EventType {
        ClientConnecting,
        ClientConnected,
        ClientDisconnected,
        UpstreamConnected,
        UpstreamDisconnected,
        DownstreamConnected,
        DownstreamDisconnected,
        FrameReceived,
        FrameSent,
        Other
    };
    
    /**
     * @brief Event structure for streaming events
     */
    struct Event {
        EventType type;
        std::string message;
        int64_t timestamp = 0;
    };
    
    /**
     * @brief Callback type for streaming events
     */
    using EventCallback = std::function<void(const Event&)>;
    
    /**
     * @brief Callback type for received frame events
     * 
     * @param frame_data Pointer to the frame data
     * @param data_size Size of the frame data in bytes
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @param format Pixel format as integer (0=BGRA, 1=BGR, 2=RGB, 3=RGBA)
     * @param timestamp Frame timestamp in milliseconds
     */
    using FrameReceivedCallback = std::function<void(const uint8_t* frame_data, size_t data_size, 
                                                    int width, int height, int format, 
                                                    uint64_t timestamp)>;
    
    /**
     * @brief Constructor for StreamingServer
     * @param config Server configuration
     */
    explicit StreamingServer(const Config& config);
    
    /**
     * @brief Destructor
     */
    ~StreamingServer();
    
    /**
     * @brief Move constructor
     */
    StreamingServer(StreamingServer&& other) noexcept = default;
    
    /**
     * @brief Move assignment operator
     */
    StreamingServer& operator=(StreamingServer&& other) noexcept = default;
    
    // Delete copy operations
    StreamingServer(const StreamingServer&) = delete;
    StreamingServer& operator=(const StreamingServer&) = delete;
    
    // Public API methods - using VideoFrame directly
    
    /**
     * @brief Start the streaming server
     * @throws std::runtime_error if server fails to start
     */
    void start();
    
    /**
     * @brief Stop the streaming server
     */
    void stop();
    
    /**
     * @brief Check if the server is running
     * @return true if server is running, false otherwise
     */
    bool isRunning() const;
    
    /**
     * @brief Send a video frame to connected clients
     * @param frame The video frame to send
     */
    void sendFrame(const VideoFrame& frame);
    
    /**
     * @brief Receive a video frame from connected clients (blocking)
     * @return The received video frame (empty if no frame available)
     */
    VideoFrame receiveFrame();
    
    /**
     * @brief Try to receive a video frame from connected clients (non-blocking)
     * @param frame Output parameter to store the received frame
     * @return true if a frame was received, false otherwise
     */
    bool tryReceiveFrame(VideoFrame& frame);
    
    /**
     * @brief Set callback for streaming events
     * @param callback The callback function to invoke on events
     */
    void setEventCallback(EventCallback callback);
    
    /**
     * @brief Set callback for received frame events
     * @param callback The callback function to invoke when frames are received
     */
    void setFrameReceivedCallback(FrameReceivedCallback callback);
    
    /**
     * @brief Check if there are connected clients
     * @return true if clients are connected, false otherwise
     */
    bool hasConnectedClients() const;
    
    /**
     * @brief Get the current server configuration
     * @return The server configuration
     */
    Config getConfig() const;
    
    /**
     * @brief Update the server configuration
     * @param config The new configuration (requires restart if server is running)
     */
    void updateConfig(const Config& config);

    // SDK callback methods - use void* for all SDK types to hide implementation
    // These are called by the SDK and forward to the implementation
    
    /**
     * @brief Handle client events (internal use)
     * @param connection Opaque connection handle
     * @param event Opaque event data
     */
    void handleClientEvent(void* connection, void* event);
    
    /**
     * @brief Handle stream events (internal use)
     * @param connection Opaque connection handle
     * @param streamConnection Opaque stream connection handle
     * @param stream Opaque stream handle
     * @param config Opaque configuration data
     * @param event Opaque event data
     */
    void handleStreamEvent(void* connection, void* streamConnection, 
                          void* stream, const void* config, void* event);
    
    /**
     * @brief Handle received video frames (internal use)
     * @param stream Opaque stream handle
     * @param frameData Opaque frame data
     */
    void handleVideoFrameReceived(void* stream, void* frameData);

private:
    // PIMPL idiom - hide all implementation details
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
