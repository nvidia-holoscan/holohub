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

#ifndef NV_VIDEO_ENCODER_NV_VIDEO_ENCODER_HPP
#define NV_VIDEO_ENCODER_NV_VIDEO_ENCODER_HPP

#include <memory>
#include <string>

#include <cuda.h>
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include "NvEncoder/NvEncoderCuda.h"
#include "nvEncodeAPI.h"

namespace holoscan::ops {

/**
 * @brief Operator to encode video frames using NVIDIA Video Codec SDK
 *
 * This operator takes video frames as input and encodes them to H264 format.
 * The input and output data remain on the GPU for maximum performance.
 */
class NvVideoEncoderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NvVideoEncoderOp)

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<uint32_t> cuda_device_ordinal_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<std::string> preset_;
  Parameter<std::string> codec_;
  Parameter<uint32_t> bitrate_;
  Parameter<uint32_t> frame_rate_;
  Parameter<uint32_t> rate_control_mode_;
  Parameter<uint32_t> multi_pass_encoding_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_;
  std::unique_ptr<NvEncoderCuda> encoder_;

  static const std::map<std::string, GUID> CODEC_GUIDS;
  static const std::map<std::string, GUID> PRESET_GUIDS;
};

}  // namespace holoscan::ops

#endif /* NV_VIDEO_ENCODER_NV_VIDEO_ENCODER_HPP */
