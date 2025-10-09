/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, DELTACAST.TV. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEOMASTER_TRANSMITTER_HPP
#define HOLOSCAN_OPERATORS_VIDEOMASTER_TRANSMITTER_HPP

#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "videomaster_base.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from Deltacast capture card.
 */
class VideoMasterTransmitterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoMasterTransmitterOp)

  VideoMasterTransmitterOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  bool configure_board_for_overlay();
  bool configure_stream_for_overlay();

  Parameter<holoscan::IOSpec*> _source;
  Parameter<bool> _use_rdma;
  Parameter<uint32_t> _board_index;
  Parameter<uint32_t> _channel_index;
  Parameter<uint32_t> _width;
  Parameter<uint32_t> _height;
  Parameter<bool> _progressive;
  Parameter<uint32_t> _framerate;
  Parameter<bool> _overlay;

  bool _has_lost_signal;
  uint64_t _slot_count;

  std::unique_ptr<VideoMasterBase> _video_master_base;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_VIDEOMASTER_TRANSMITTER_HPP */
