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

#ifndef OPERATORS_SPLIT_VIDEO
#define OPERATORS_SPLIT_VIDEO

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

namespace holoscan::ops {
#define STEREO_VIDEO_HORIZONTAL 0
#define STEREO_VIDEO_VERTICAL 1

// splits a horizontally or vertically stacked stereo video into two video streams
class SplitVideoOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SplitVideoOp);
  SplitVideoOp() = default;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<int> stereo_video_layout_;
};

// combines two videos into a single horizontally or vertically stacked video
class MergeVideoOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MergeVideoOp);
  MergeVideoOp() = default;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<int> stereo_video_layout_;
};
}  // namespace holoscan::ops
#endif
