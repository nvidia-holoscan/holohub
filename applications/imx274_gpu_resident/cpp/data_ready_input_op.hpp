/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef HOLOHUB_IMX274_GPU_RESIDENT_DATA_READY_INPUT_OP
#define HOLOHUB_IMX274_GPU_RESIDENT_DATA_READY_INPUT_OP

#include <memory>

#include <holoscan/core/gpu_resident_operator.hpp>

#include "shared_frame_state.hpp"

namespace imx274_gpu_resident {

class DataReadyInputOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DataReadyInputOp, holoscan::GPUResidentOperator);

  DataReadyInputOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override;

  void set_shared_state(std::shared_ptr<SharedFrameState> shared_state);

 private:
  unsigned char** chosen_frame_memory_ = nullptr;
  CUdeviceptr frame_memory_ = 0;
  size_t frame_size_rounded_ = 0;
  std::shared_ptr<SharedFrameState> shared_state_;
};

}  // namespace imx274_gpu_resident

#endif /* HOLOHUB_IMX274_GPU_RESIDENT_DATA_READY_INPUT_OP */
