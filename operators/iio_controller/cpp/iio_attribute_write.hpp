/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Analog Devices, Inc. All rights reserved.
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

#include <iio.h>
#include <holoscan/core/forward_def.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/logger/logger.hpp>
#include "iio_params.hpp"

namespace holoscan::ops {

class IIOAttributeWrite : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IIOAttributeWrite)

  IIOAttributeWrite() = default;
  ~IIOAttributeWrite() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& ec) override;

 private:
  Parameter<std::string> ctx_p_;
  Parameter<std::string> dev_p_;
  Parameter<std::string> chan_p_;
  Parameter<bool> channel_is_output_;
  Parameter<std::string> attr_name_p_;

  iio_context* ctx_;
  iio_device* dev_;
  iio_channel* chan_;
  std::string attr_name_;
  attr_type_t attr_type_;

  char buffer[1024];
  ssize_t ret = -1;

  // Error flags for initialization failures
  bool ctx_creation_failed_ = false;
  bool dev_not_found_ = false;
  bool chan_not_found_ = false;
};

}  // namespace holoscan::ops
