/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_AJA_SOURCE_AJA_SOURCE_HPP
#define HOLOSCAN_OPERATORS_AJA_SOURCE_AJA_SOURCE_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
// AJA headers are not clean C++ and generate warnings
// when compiled with -Werror.
// Since ajantv2 is 3rd party code, we disable the warning for now.
#include <ajantv2/includes/ntv2card.h>
#include <ajantv2/includes/ntv2devicescanner.h>
#pragma GCC diagnostic pop
#include <ajantv2/includes/ntv2enums.h>

#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "./ntv2channel.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from AJA capture card.
 *
 * ==Named Inputs==
 *
 * - **overlay_buffer_input** : `nvidia::gxf::VideoBuffer` (optional)
 *   - The operator does not require a message on this input port in order for `compute` to
 *     be called. If a message is found, and `enable_overlay` is true, the image will be
 *     mixed with the image captured by the AJA card. If `enable_overlay` is false, any message
 *     on this port will be ignored.
 *
 * ==Named Outputs==
 *
 * - **video_buffer_output** : `nvidia::gxf::VideoBuffer`
 *   - The output video frame from the AJA capture card. If `overlay_rdma` is true, this
 *     video buffer will be on the device, otherwise it will be in pinned host memory.
 * - **overlay_buffer_output** : `nvidia::gxf::VideoBuffer` (optional)
 *   - This output port will only emit a video buffer when `enable_overlay` is true. If
 *     `overlay_rdma` is true, this video buffer will be on the device, otherwise it will be
 *     in pinned host memory.
 *
 * ==Parameters==
 *
 * - **device**: The device to target (e.g., "0" for device 0). Optional (default: "0").
 * - **channel**: The camera `NTV2Channel` to use for output (e.g., `NTV2Channel::NTV2_CHANNEL1`
 *   (`0`) or "NTV2_CHANNEL1" (in YAML) for the first channel). Optional (default:
 *   `NTV2Channel::NTV2_CHANNEL1` in C++ or `"NTV2_CHANNEL1"` in YAML).
 * - **width**: Width of the video stream. Optional (default: `1920`).
 * - **height**: Height of the video stream. Optional (default: `1080`).
 * - **framerate**: Frame rate of the video stream. Optional (default: `60`).
 * - **interlaced**: Whether the frame is interlaced (true) or progressive (false). Optional (default: `false`).
 * - **rdma**: Boolean indicating whether RDMA is enabled. Optional (default: `false`).
 * - **enable_overlay**: Boolean indicating whether a separate overlay channel is enabled. Optional
 *   (default: `false`).
 * - **overlay_channel**: The camera `NTV2Channel` to use for overlay output. Optional (default:
 *   `NTV2Channel::NTV2_CHANNEL2` in C++ or `"NTV2_CHANNEL2"` in YAML).
 * - **overlay_rdma**: Boolean indicating whether RDMA is enabled for the overlay. Optional
 *   (default: `true`).
 */
class AJASourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AJASourceOp)

  AJASourceOp();

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  AJAStatus DetermineVideoFormat();
  AJAStatus OpenDevice();
  AJAStatus SetupVideo();
  AJAStatus SetupBuffers();
  AJAStatus StartAutoCirculate();
  bool AllocateBuffers(std::vector<void*>& buffers, size_t num_buffers, size_t buffer_size,
                       bool rdma);
  void FreeBuffers(std::vector<void*>& buffers, bool rdma);
  bool GetNTV2VideoFormatTSI(NTV2VideoFormat* format);

  Parameter<holoscan::IOSpec*> video_buffer_output_;
  Parameter<std::string> device_specifier_;
  Parameter<NTV2Channel> channel_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> framerate_;
  Parameter<bool> interlaced_;
  Parameter<bool> use_rdma_;
  Parameter<bool> enable_overlay_;
  Parameter<NTV2Channel> overlay_channel_;
  Parameter<bool> overlay_rdma_;
  Parameter<holoscan::IOSpec*> overlay_buffer_input_;
  Parameter<holoscan::IOSpec*> overlay_buffer_output_;

  // internal state
  CNTV2Card device_;
  NTV2DeviceID device_id_ = DEVICE_ID_NOTFOUND;
  NTV2VideoFormat video_format_ = NTV2_FORMAT_UNKNOWN;
  NTV2PixelFormat pixel_format_ = NTV2_FBF_ABGR;
  bool use_tsi_ = false;
  uint8_t hdmi_in_count_ = 0;

  std::vector<void*> buffers_;
  std::vector<void*> overlay_buffers_;
  uint8_t current_buffer_ = 0;
  uint8_t current_hw_frame_ = 0;
  uint8_t current_overlay_hw_frame_ = 0;

  bool is_igpu_ = false;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_AJA_SOURCE_AJA_SOURCE_HPP */
