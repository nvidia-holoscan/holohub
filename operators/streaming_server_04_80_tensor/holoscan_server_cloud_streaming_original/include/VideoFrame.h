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

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

enum class PixelFormat {
    YUV420,
    RGBA,
    NV12,
    BGRA,
    BGR  // Add BGR format for 3-channel images
};

class VideoFrame {
public:
    // Default constructor with safe defaults
    VideoFrame() : m_width(1), m_height(1), m_timestamp(0), m_format(PixelFormat::BGR) {
        // Initialize with at least 3 bytes (one BGR pixel)
        m_dataBuffer.resize(3, 0);
        m_data = m_dataBuffer.data();
    }
    
    // Full constructor
    VideoFrame(uint32_t width, uint32_t height, const uint8_t* data, size_t size, uint64_t timestamp)
        : m_width(width > 0 ? width : 1),
          m_height(height > 0 ? height : 1),
          m_timestamp(timestamp),
          m_format(PixelFormat::BGR) {
        setData(data, size);
    }
    
    // Copy constructor
    VideoFrame(const VideoFrame& other)
        : m_width(other.m_width),
          m_height(other.m_height),
          m_timestamp(other.m_timestamp),
          m_format(other.m_format),
          m_dataBuffer(other.m_dataBuffer) {
        m_data = m_dataBuffer.empty() ? nullptr : m_dataBuffer.data();
    }
    
    // Constructor for pre-allocated frames
    VideoFrame(uint32_t width, uint32_t height) 
        : m_width(width > 0 ? width : 1),
          m_height(height > 0 ? height : 1),
          m_timestamp(0),
          m_format(PixelFormat::BGR) {
        // Allocate data buffer of appropriate size for BGR: 3 bytes per pixel
        m_dataBuffer.resize(m_width * m_height * 3); // BGR: 3 bytes per pixel
        m_data = m_dataBuffer.data();
    }
    
    // Setter methods
    void setWidth(uint32_t width) {
        m_width = width > 0 ? width : 1; // Never allow zero width
    }
    
    void setHeight(uint32_t height) {
        m_height = height > 0 ? height : 1; // Never allow zero height
    }
    
    void setData(const uint8_t* data, size_t size) {
        if (data && size > 0) {
            m_dataBuffer.resize(size);
            std::memcpy(m_dataBuffer.data(), data, size);
            m_data = m_dataBuffer.data();
        } else {
            // Create a minimal valid buffer (1 pixel)
            m_dataBuffer.resize(4, 0);
            m_data = m_dataBuffer.data();
        }
    }
    
    void setTimestamp(uint64_t timestamp) { m_timestamp = timestamp; }
    
    // Getter methods
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    const uint8_t* getData() const { return m_data; }
    size_t getDataSize() const { return m_dataBuffer.size(); }
    uint64_t getTimestamp() const { return m_timestamp; }
    
    // Check if frame contains valid data
    bool isValid() const {
        return m_width > 0 && m_height > 0 && !m_dataBuffer.empty(); 
    }
    
    // Get writable data buffer - ensure it's properly sized first
    uint8_t* getWritableData() {
        size_t required_size = m_width * m_height * getBytesPerPixel();
        if (m_dataBuffer.empty() || m_dataBuffer.size() < required_size) {
            m_dataBuffer.resize(required_size);
            m_data = m_dataBuffer.data();
        }
        return m_dataBuffer.data();
    }
    
    // Format handling
    PixelFormat getFormat() const {
        return m_format;
    }

    void setFormat(PixelFormat format) {
        m_format = format;
    }
    
    // Get bytes per pixel for the current format
    size_t getBytesPerPixel() const {
        switch (m_format) {
            case PixelFormat::BGR:
                return 3;
            case PixelFormat::RGBA:
            case PixelFormat::BGRA:
                return 4;
            case PixelFormat::YUV420:
                return 1; // Y component only
            case PixelFormat::NV12:
                return 1; // Y component only
            default:
                return 3; // Default to BGR
        }
    }

private:
    uint32_t m_width;
    uint32_t m_height;
    std::vector<uint8_t> m_dataBuffer;  // Owns the data
    const uint8_t* m_data;              // Points to our data buffer
    uint64_t m_timestamp;
    PixelFormat m_format = PixelFormat::BGR; // Default format
};

// Define the frame generator function type globally
using FrameGeneratorFunc = std::function<VideoFrame()>;
