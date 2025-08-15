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



/**
 * @brief Frame structure that abstracts away SDK-specific image buffer details
 */
struct Frame {
    std::vector<uint8_t> data;  // Frame data
    uint32_t width;             // Frame width
    uint32_t height;            // Frame height
    int64_t timestampMs;         // Timestamp in milliseconds
    
    // Frame format enum that replaces SDK-specific format
    enum class Format {
        BGRA,
        RGBA,
        BGR,
        NV12,
        // Add other formats as needed
    };
    Format format = Format::BGR;  // Default to BGR format for client compatibility
};

/**
 * @brief StreamingServer class that unifies all StreamSDK functionality
 */
class StreamingServer
{
public:
    /**
     * @brief Configuration structure for server settings
     */
    struct Config {
        // Server settings
        uint16_t port = 48010;     // Server port (updated to new default)
        bool isMultiInstance = false; // Allow multiple server instances
        std::string serverName = "StreamingServer"; // Name identifier for the server
        
        // Video settings
        uint32_t width = 854;      // Video width
        uint32_t height = 480;     // Video height
        uint16_t fps = 30;         // Video frame rate
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
    
    // Public API methods
    void start();
    void stop();
    bool isRunning() const;
    void sendFrame(const Frame& frame);
    Frame receiveFrame();
    bool tryReceiveFrame(Frame& frame);
    void setEventCallback(EventCallback callback);
    bool hasConnectedClients() const;
    Config getConfig() const;
    void updateConfig(const Config& config);

    // SDK callback methods - use void* for all SDK types
    // to avoid redeclaring SDK types in the header
    void handleClientEvent(void* connection, void* event);
    void handleStreamEvent(void* connection, void* streamConnection, 
                          void* stream, const void* config, void* event);
    void handleVideoFrameReceived(void* stream, void* frameData);

private:
    // PIMPL idiom
    class Impl;
    std::unique_ptr<Impl> pImpl;
};