// Copyright 2025 Analog Devices, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iio.h>
#include <holoscan/core/forward_def.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan::ops {

class IIOBufferWrite : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IIOBufferWrite);

  IIOBufferWrite() = default;
  ~IIOBufferWrite() = default;

  void stop() override;
  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& ec) override;

 private:
  Parameter<std::string> ctx_p_;
  Parameter<std::string> dev_p_;
  Parameter<std::vector<std::string>> enabled_channel_names_p_;
  Parameter<std::vector<bool>> enabled_channel_types_p_;
  Parameter<bool> is_cyclic;

  iio_context* ctx_ = nullptr;
  iio_device* dev_ = nullptr;
  iio_buffer* buffer_ = nullptr;
  uint32_t buffer_samples_count_ = 0;
  ssize_t sample_size_ = 0;

  // Error flags for initialization failures
  bool channels_empty_ = false;
  bool ctx_creation_failed_ = false;
  bool dev_not_found_ = false;
  bool chan_not_found_ = false;
  bool sample_size_failed_ = false;
};

}  // namespace holoscan::ops
