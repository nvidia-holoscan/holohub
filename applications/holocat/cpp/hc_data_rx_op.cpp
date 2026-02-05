/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

/**
 * @file hc_data_rx_op.cpp
 * @brief HoloCat Data Receive Operator
 *
 * HoloCat Data Receive Operator
 *
 * This file implements the HcDataRxOp operator that receives counter data
 * for testing and demonstration purposes.
 */

#include <holoscan/holoscan.hpp>

#include "hc_data_rx_op.hpp"

namespace holocat {

void HcDataRxOp::setup(holoscan::OperatorSpec& spec) {
  // Configure input port for receiving counter data
  spec.input<int>("count_in");

  HOLOSCAN_LOG_INFO("HcDataRxOp: Setup complete - configured input port 'count_in'");
}

void HcDataRxOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                         holoscan::ExecutionContext& context) {
  // Receive count value from ECat bus
  auto maybe_count = op_input.receive<int>("count_in");
  if (!maybe_count) {
    HOLOSCAN_LOG_ERROR("HcDataRxOp: Failed to receive count from ECat bus");
    return;
  }
  last_count_ = maybe_count.value();
  HOLOSCAN_LOG_INFO("HcDataRxOp: Received count: {}", last_count_);
}

}  // namespace holocat

/*-END OF SOURCE FILE--------------------------------------------------------*/
