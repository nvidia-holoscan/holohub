/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_EMERGENT_SOURCE_HPP
#define HOLOSCAN_OPERATORS_EMERGENT_SOURCE_HPP

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to get the video stream from Emergent Vision Technologies
 * camera using MLNX ConnectX SmartNIC.
 *
 * This wraps a GXF Codelet(`nvidia::holoscan::EmergentSource`).
 */
class EmergentSourceOp : public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(EmergentSourceOp, holoscan::ops::GXFOperator)

  EmergentSourceOp() = default;

  const char* gxf_typename() const override { return "nvidia::holoscan::EmergentSource"; }

  void setup(OperatorSpec& spec) override;

  void initialize() override;

 private:
  Parameter<holoscan::IOSpec*> signal_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> framerate_;
  Parameter<bool> use_rdma_;
  Parameter<uint32_t> exposure_;
  Parameter<uint32_t> gain_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_EMERGENT_SOURCE_HPP */

