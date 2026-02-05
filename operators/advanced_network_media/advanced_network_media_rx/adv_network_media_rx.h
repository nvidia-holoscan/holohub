/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_ADV_NETWORK_MEDIA_RX_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_ADV_NETWORK_MEDIA_RX_H_

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

// Forward declare the implementation class
class AdvNetworkMediaRxOpImpl;

/**
 * @class AdvNetworkMediaRxOp
 * @brief Operator for receiving media frames over advanced network infrastructure.
 *
 * This operator receives video frames over Rivermax-enabled network infrastructure
 * and outputs them as GXF VideoBuffer entities.
 */
class AdvNetworkMediaRxOp : public Operator {
 public:
  static constexpr uint16_t default_queue_id = 0;

  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkMediaRxOp)

  /**
   * @brief Constructs an AdvNetworkMediaRxOp operator.
   */
  AdvNetworkMediaRxOp();

  /**
   * @brief Destroys the AdvNetworkMediaRxOp operator and its implementation.
   */
  ~AdvNetworkMediaRxOp();

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
    ExecutionContext& context) override;

 protected:
  // Operator parameters
  Parameter<std::string> interface_name_;
  Parameter<uint16_t> queue_id_;
  Parameter<uint32_t> frame_width_;
  Parameter<uint32_t> frame_height_;
  Parameter<uint32_t> bit_depth_;
  Parameter<std::string> video_format_;
  Parameter<bool> hds_;
  Parameter<std::string> output_format_;
  Parameter<std::string> memory_location_;

 private:
  friend class AdvNetworkMediaRxOpImpl;

  std::unique_ptr<AdvNetworkMediaRxOpImpl> pimpl_;
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_ADV_NETWORK_MEDIA_RX_H_
