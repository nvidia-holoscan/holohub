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

namespace holoscan::ops {

class IIOConfigurator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IIOConfigurator);

  IIOConfigurator() = default;
  ~IIOConfigurator() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& ec) override;

  enum class IIODeviceAttrType { DEVICE, DEBUG, BUFFER };

 private:
  /* YAML IIO context navigation functions */
  void parse_setup(const YAML::Node& setup_node, iio_context* ctx);

  void parse_device(const YAML::Node& device_node, iio_device* dev);
  void parse_attribute(const YAML::Node& attr_node, iio_device* dev,
                       IIODeviceAttrType type = IIODeviceAttrType::DEVICE);

  void parse_channel(const YAML::Node& channel_node, iio_channel* chan);
  void parse_attribute(const YAML::Node& attr_node, iio_channel* chan);
  Parameter<std::string> cfg_path_p_;
};

}  // namespace holoscan::ops
