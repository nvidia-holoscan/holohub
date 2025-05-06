/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 DELTACAST.TV. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEOMASTER_SOURCE_HPP
#define HOLOSCAN_OPERATORS_VIDEOMASTER_SOURCE_HPP

#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from Deltacast capture card.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::videomaster::VideoMasterSource`).
 */
class VideoMasterSourceOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(VideoMasterSourceOp, holoscan::ops::GXFOperator)

  VideoMasterSourceOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::holoscan::videomaster::VideoMasterSource";
  }

  void setup(OperatorSpec& spec) override;

 private:
  Parameter<holoscan::IOSpec*> _signal;
  Parameter<bool> _use_rdma;
  Parameter<uint32_t> _board_index;
  Parameter<uint32_t> _channel_index;
  Parameter<std::shared_ptr<Allocator>> _pool;
  Parameter<uint32_t> _width;
  Parameter<uint32_t> _height;
  Parameter<bool> _progressive;
  Parameter<uint32_t> _framerate;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VIDEOMASTER_SOURCE_HPP */
