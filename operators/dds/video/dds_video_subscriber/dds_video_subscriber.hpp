/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <dds/sub/ddssub.hpp>

#include "dds_operator_base.hpp"
#include "VideoFrame.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to subscribe to a DDS video stream.
 */
class DDSVideoSubscriberOp : public DDSOperatorBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSVideoSubscriberOp, DDSOperatorBase)

  DDSVideoSubscriberOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> reader_qos_;
  Parameter<uint32_t> stream_id_;

  dds::sub::DataReader<VideoFrame> reader_ = dds::core::null;
  dds::core::cond::StatusCondition status_condition_ = dds::core::null;
  dds::core::cond::WaitSet waitset_;
};

}  // namespace holoscan::ops
