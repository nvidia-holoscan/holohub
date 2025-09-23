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

#include "VideoFrame.h"
#include <chrono>
#include <memory>
#include <string>
#include <functional>

// Forward declaration of implementation class
class StreamingClientImpl;

// FrameGeneratorFunc is already defined in VideoFrame.h

/**
 * @brief A unified client that handles both upstream and downstream video streams with MANUAL frame sending
 * 
 * This class provides manual frame sending using sendFrame() for precise control over when frames are sent.
 * Use StreamingClientAutomatic for automatic frame generation.
 */
class StreamingClient {
public:
    // Define types for callbacks and frame generation
    using FrameCallback = std::function<void(const VideoFrame&)>;
    
    /**
     * @brief Construct a new streaming client
     * 
     * @param width Width of video frames (default: 854)
     * @param height Height of video frames (default: 480)
     * @param fps Frames per second (default: 30)
     * @param signalingPort Port used for signaling (default: 48010)
     */
    StreamingClient(uint32_t width = 854, uint32_t height = 480, uint32_t fps = 30, uint16_t signalingPort = 48010);
    
    /**
     * @brief Destroy the client and clean up resources
     */
    ~StreamingClient();
    
    // Move operations
    StreamingClient(StreamingClient&&) noexcept;
    StreamingClient& operator=(StreamingClient&&) noexcept;
    
    // Delete copy operations
    StreamingClient(const StreamingClient&) = delete;
    StreamingClient& operator=(const StreamingClient&) = delete;
    
    /**
     * @brief Start streaming to/from the server
     * 
     * @param serverIp IP address of the server
     * @param signalingPort Port used for signaling
     */
    void startStreaming(const std::string& serverIp, uint16_t signalingPort);
    
    /**
     * @brief Stop all streaming
     */
    void stopStreaming();
    
    /**
     * @brief Wait for streaming to end
     * 
     * @param timeout Maximum time to wait
     * @return true if streaming ended, false if timeout occurred
     */
    bool waitForStreamingEnded(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
    
    /**
     * @brief Wait for first video frame to be received
     * 
     * @param timeout Maximum time to wait
     * @return true if first frame was received, false if timeout occurred
     */
    bool waitForFirstFrameReceived(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
    
    /**
     * @brief Set a callback function for received frames
     * 
     * @param callback Function to be called when a new frame is received
     */
    void setFrameCallback(FrameCallback callback);
    
    /**
     * @brief Set a callback function for received frames (alias for setFrameCallback)
     */
    void setFrameReceivedCallback(FrameCallback callback);
    
    /**
     * @brief Check if the client is currently streaming
     * 
     * @return true if streaming, false otherwise
     */
    bool isStreaming() const;
    
    /**
     * @brief Check if the upstream video connection is ready for sending frames
     * 
     * @return true if upstream is ready, false otherwise
     */
    bool isUpstreamReady() const;
    
    /**
     * @brief Explicitly clean up resources before destruction
     * 
     * Call this before the application exits to ensure clean shutdown.
     */
    void cleanup();
    
    /**
     * @brief Send a single frame to the server
     * 
     * This method directly sends the provided frame to the server without requiring a frame source.
     * It encapsulates the logic from ClientUpstreamVideo::onConnected for manual frame sending.
     * 
     * @param frame The video frame to send
     * @return bool true if the frame was sent successfully, false otherwise
     */
    bool sendFrame(const VideoFrame& frame);
    
private:
    // Private implementation pointer
    std::unique_ptr<StreamingClientImpl> pImpl;
};
