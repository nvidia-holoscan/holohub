/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEO_READ_BITSTREAM_VIDEO_READ_BITSTREAM
#define HOLOSCAN_OPERATORS_VIDEO_READ_BITSTREAM_VIDEO_READ_BITSTREAM

#include <memory>
#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to read video bit stream.
 *
 * This wraps a GXF Codelet(`nvidia::gxf::VideoReadBitStream`).
 */
class VideoReadBitstreamOp: public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(VideoReadBitstreamOp, holoscan::ops::GXFOperator)


  VideoReadBitstreamOp() = default;
  const char* gxf_typename() const override {
    return "nvidia::gxf::VideoReadBitStream";
  }

  void setup(OperatorSpec& spec) override;

 private:
  Parameter<holoscan::IOSpec*> output_transmitter_;
  Parameter<std::string> input_file_path_;
  Parameter<std::shared_ptr<Allocator>> pool_;
  Parameter<int32_t> outbuf_storage_type_;
  Parameter<int32_t> aud_nal_present_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_READ_BITSTREAM_VIDEO_READ_BITSTREAM
