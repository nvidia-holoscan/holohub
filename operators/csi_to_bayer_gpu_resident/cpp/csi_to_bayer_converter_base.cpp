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

#include <csi_to_bayer_gpu_resident/csi_to_bayer_converter_base.hpp>

#include <holoscan/logger/logger.hpp>
#include <hololink/core/networking.hpp>

#include <fmt/format.h>
#include <stdexcept>

namespace hololink::operators {

uint32_t CsiToBayerConverterBase::receiver_start_byte() {
    return 0;
}

uint32_t CsiToBayerConverterBase::received_line_bytes(
    uint32_t transmitted_line_bytes) {
    return hololink::core::round_up(transmitted_line_bytes, 8);
}

uint32_t CsiToBayerConverterBase::transmitted_line_bytes(
    hololink::csi::PixelFormat pixel_format,
    uint32_t pixel_width) {
    switch (pixel_format) {
    case hololink::csi::PixelFormat::RAW_8:
        return pixel_width;
    case hololink::csi::PixelFormat::RAW_10:
        return pixel_width * 5 / 4;
    case hololink::csi::PixelFormat::RAW_12:
        return pixel_width * 3 / 2;
    default:
        throw std::runtime_error(
            fmt::format("Unsupported pixel format {}",
                        int(pixel_format)));
    }
}

void CsiToBayerConverterBase::configure(
    uint32_t start_byte, uint32_t bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    hololink::csi::PixelFormat pixel_format,
    uint32_t trailing_bytes) {
    HOLOSCAN_LOG_INFO(
        "start_byte={}, bytes_per_line={}, pixel_width={}, "
        "pixel_height={}, pixel_format={}, trailing_bytes={}.",
        start_byte, bytes_per_line, pixel_width, pixel_height,
        static_cast<int>(pixel_format), trailing_bytes);
    start_byte_ = start_byte;
    bytes_per_line_ = bytes_per_line;
    pixel_width_ = pixel_width;
    pixel_height_ = pixel_height;
    pixel_format_ = pixel_format;
    csi_length_ = start_byte + bytes_per_line * pixel_height
                  + trailing_bytes;
    frame_size_ = csi_length_;
    configured_ = true;
}

size_t CsiToBayerConverterBase::get_csi_length() const {
    if (!configured_) {
        throw std::runtime_error(
            "CsiToBayer converter is not configured.");
    }
    return csi_length_;
}

size_t CsiToBayerConverterBase::get_frame_size() const {
    if (!configured_) {
        throw std::runtime_error(
            "CsiToBayer converter is not configured.");
    }
    return frame_size_;
}

}  // namespace hololink::operators
