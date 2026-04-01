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
#ifndef HOLOHUB_CSI_TO_BAYER_GPU_RESIDENT_CSI_TO_BAYER_GPU_RESIDENT
#define HOLOHUB_CSI_TO_BAYER_GPU_RESIDENT_CSI_TO_BAYER_GPU_RESIDENT

#include <memory>
#include "csi_to_bayer_converter_base.hpp"

#include <holoscan/core/gpu_resident_operator.hpp>

#include <cuda.h>

namespace hololink::common {
class CudaFunctionLauncher;
}  // namespace hololink::common

namespace hololink::operators {

class CsiToBayerGpuResidentOp : public holoscan::GPUResidentOperator,
                                public CsiToBayerConverterBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(CsiToBayerGpuResidentOp, holoscan::GPUResidentOperator);

  CsiToBayerGpuResidentOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override;

 private:
  CUcontext cuda_context_ = nullptr;
  CUdevice cuda_device_ = 0;

  std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;
};

}  // namespace hololink::operators

#endif /* HOLOHUB_CSI_TO_BAYER_GPU_RESIDENT_CSI_TO_BAYER_GPU_RESIDENT */
