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
 * @file hc_data_tx_op.cpp
 * @brief HoloCat Data Transmit Operator Implementation
 *
 * This file implements the HcDataTxOp operator that generates incrementing
 * counter data for testing and demonstration purposes.
 */

#include <holoscan/holoscan.hpp>

#include "hc_data_tx_op.hpp"

namespace holocat {

void HcDataTxOp::setup(holoscan::OperatorSpec& spec) {
  // Configure output port for emitting counter data
  spec.output<int>("count_out");

  HOLOSCAN_LOG_INFO("HcDataTxOp: Setup complete - configured output port 'count_out'");
}

void HcDataTxOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                         holoscan::ExecutionContext& context) {
  counter_ = (counter_ + 1) % kMaxCount;
  op_output.emit<int>(counter_, "count_out");
  HOLOSCAN_LOG_DEBUG("HcDataTxOp: 50x Emitted count = {}", counter_);
}

}  // namespace holocat
/*-END OF SOURCE FILE--------------------------------------------------------*/
