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

#include "frame_debug_utils.hpp"

#ifdef HOLOSCAN_DEBUG_FRAME_WRITING

#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstdint>

#include <holoscan/holoscan.hpp>

// Include StreamingServer header to get Frame class definition
#include "streaming_server_resource.hpp"

namespace holoscan::ops::debug_utils {

bool writeFrameToDisk(const Frame& frame, const std::string& filename_prefix, int frame_number) {
    try {
        // Generate filename with timestamp and frame number
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream filename;
        filename << filename_prefix;
        if (frame_number >= 0) {
            filename << "_frame" << std::setfill('0') << std::setw(6) << frame_number;
        }
        filename << "_" << std::put_time(std::localtime(&time_t), "%H%M%S")
                 << "_" << std::setfill('0') << std::setw(3) << ms.count();

        // Get frame properties
        uint32_t width = frame.getWidth();
        uint32_t height = frame.getHeight();
        const uint8_t* data = frame.getData();
        size_t data_size = frame.getDataSize();
        auto format = frame.getFormat();

        if (!data || data_size == 0 || width == 0 || height == 0) {
            HOLOSCAN_LOG_ERROR("writeFrameToDisk: Invalid frame data");
            return false;
        }

        // Write raw binary data
        std::string raw_filename = filename.str() + "_raw.bin";
        std::ofstream raw_file(raw_filename, std::ios::binary);
        if (raw_file.is_open()) {
            raw_file.write(reinterpret_cast<const char*>(data), data_size);
            raw_file.close();
            HOLOSCAN_LOG_INFO("Wrote raw frame data to: {}", raw_filename);
        }

        // Write metadata file
        std::string meta_filename = filename.str() + "_meta.txt";
        std::ofstream meta_file(meta_filename);
        if (meta_file.is_open()) {
            meta_file << "Frame Metadata:\n";
            meta_file << "Width: " << width << "\n";
            meta_file << "Height: " << height << "\n";
            meta_file << "Data Size: " << data_size << " bytes\n";
            meta_file << "Pixel Format: " << static_cast<int>(format) << " (";
            switch (format) {
                case ::PixelFormat::BGR: meta_file << "BGR"; break;
                case ::PixelFormat::BGRA: meta_file << "BGRA"; break;
                case ::PixelFormat::RGBA: meta_file << "RGBA"; break;
                default: meta_file << "Unknown"; break;
            }
            meta_file << ")\n";
            meta_file << "Timestamp: " << frame.getTimestamp() << "\n";
            meta_file << "Bytes per pixel: " << (data_size / (width * height)) << "\n";

            // Add first few pixel values for inspection
            meta_file << "\nFirst 10 pixels (raw bytes):\n";
            size_t bytes_per_pixel =
                (format == ::PixelFormat::BGRA || format == ::PixelFormat::RGBA) ? 4 : 3;
            for (int i = 0; i < std::min(10, static_cast<int>(width * height)) &&
                            i * bytes_per_pixel < data_size; ++i) {
                meta_file << "Pixel " << i << ": ";
                for (size_t j = 0; j < bytes_per_pixel &&
                                   (i * bytes_per_pixel + j) < data_size; ++j) {
                    meta_file << static_cast<int>(data[i * bytes_per_pixel + j]) << " ";
                }
                meta_file << "\n";
            }

            meta_file.close();
            HOLOSCAN_LOG_INFO("Wrote frame metadata to: {}", meta_filename);
        }

        // Write as PPM image file (for easy viewing)
        if (format == ::PixelFormat::BGRA || format == ::PixelFormat::BGR ||
            format == ::PixelFormat::RGBA) {
            std::string ppm_filename = filename.str() + ".ppm";
            std::ofstream ppm_file(ppm_filename, std::ios::binary);
            if (ppm_file.is_open()) {
                // PPM header
                ppm_file << "P6\n" << width << " " << height << "\n255\n";

                // Convert pixel data to RGB for PPM
                size_t bytes_per_pixel =
                    (format == ::PixelFormat::BGRA || format == ::PixelFormat::RGBA) ? 4 : 3;
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        size_t pixel_offset = (y * width + x) * bytes_per_pixel;
                        if (pixel_offset + 2 < data_size) {
                            uint8_t r, g, b;
                            if (format == ::PixelFormat::BGRA || format == ::PixelFormat::BGR) {
                                // BGR(A) format - swap B and R
                                b = data[pixel_offset + 0];
                                g = data[pixel_offset + 1];
                                r = data[pixel_offset + 2];
                            } else {
                                // RGB(A) format
                                r = data[pixel_offset + 0];
                                g = data[pixel_offset + 1];
                                b = data[pixel_offset + 2];
                            }
                            ppm_file.write(reinterpret_cast<const char*>(&r), 1);
                            ppm_file.write(reinterpret_cast<const char*>(&g), 1);
                            ppm_file.write(reinterpret_cast<const char*>(&b), 1);
                        }
                    }
                }
                ppm_file.close();
                HOLOSCAN_LOG_INFO("Wrote PPM image to: {} (can be viewed with image viewers)",
                                  ppm_filename);
            }
        }

        return true;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("writeFrameToDisk exception: {}", e.what());
        return false;
    }
}

}  // namespace holoscan::ops::debug_utils

#endif  // HOLOSCAN_DEBUG_FRAME_WRITING
