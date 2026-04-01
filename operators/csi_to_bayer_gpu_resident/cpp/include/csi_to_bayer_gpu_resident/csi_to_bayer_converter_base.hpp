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

#ifndef HOLOHUB_CSI_TO_BAYER_GPU_RESIDENT_CSI_TO_BAYER_CONVERTER_BASE
#define HOLOHUB_CSI_TO_BAYER_GPU_RESIDENT_CSI_TO_BAYER_CONVERTER_BASE

#include <cstddef>

#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>

namespace hololink::operators {

/**
 * Base CSI-to-Bayer conversion state and CsiConverter implementation.
 * Used by both CsiToBayerOp and CsiToBayerGpuResidentOp to avoid code duplication.
 */
class CsiToBayerConverterBase
    : public hololink::csi::CsiConverter {
 public:
    CsiToBayerConverterBase() = default;

    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(
        hololink::csi::PixelFormat pixel_format,
        uint32_t pixel_width) override;
    void configure(
        uint32_t start_byte, uint32_t bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        hololink::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes) override;

    size_t get_csi_length() const;
    size_t get_frame_size() const;

 protected:
    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::csi::PixelFormat pixel_format_ =
        hololink::csi::PixelFormat::RAW_8;
    uint32_t start_byte_ = 0;
    uint32_t bytes_per_line_ = 0;
    size_t csi_length_ = 0;
    size_t frame_size_ = 0;
    bool configured_ = false;
};

}  // namespace hololink::operators

#endif /* HOLOHUB_CSI_TO_BAYER_GPU_RESIDENT_CSI_TO_BAYER_CONVERTER_BASE */
