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

#include "holoscan/holoscan.hpp"

using holoscan::OperatorSpec;
using holoscan::InputContext;
using holoscan::OutputContext;
using holoscan::ExecutionContext;
using holoscan::Arg;
using holoscan::ArgList;

#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "videomaster_base.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from Deltacast capture card.
 *
 */
class VideoMasterSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoMasterSourceOp)

  VideoMasterSourceOp();

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  void transmit_buffer_data(void* buffer, uint32_t buffer_size, OutputContext& op_output, ExecutionContext& context);

  Parameter<holoscan::IOSpec*> _signal;
  Parameter<bool> _use_rdma;
  Parameter<uint32_t> _board_index;
  Parameter<uint32_t> _channel_index;
  Parameter<uint32_t> _width;
  Parameter<uint32_t> _height;
  Parameter<bool> _progressive;
  Parameter<uint32_t> _framerate;

  bool _has_lost_signal;
  
  VideoMasterBase _video_master_base;
  
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VIDEOMASTER_SOURCE_HPP */
