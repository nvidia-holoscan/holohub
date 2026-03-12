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
#ifndef HOLOHUB_IMAGE_PROCESSOR_GPU_RESIDENT_IMAGE_PROCESSOR_GPU_RESIDENT
#define HOLOHUB_IMAGE_PROCESSOR_GPU_RESIDENT_IMAGE_PROCESSOR_GPU_RESIDENT

#include <memory>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/core/parameter.hpp>

#include <hololink/common/cuda_helper.hpp>

namespace hololink::common {
class CudaFunctionLauncher;
}  // namespace hololink::common

namespace hololink::operators {

class ImageProcessorGpuResidentOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ImageProcessorGpuResidentOp, holoscan::GPUResidentOperator);

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override;

 private:
  holoscan::Parameter<int> pixel_format_;
  holoscan::Parameter<int> bayer_format_;
  holoscan::Parameter<int32_t> optical_black_;
  holoscan::Parameter<int32_t> width_;
  holoscan::Parameter<int32_t> height_;

  CUcontext cuda_context_ = nullptr;
  CUdevice cuda_device_ = 0;

  std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;

  hololink::common::UniqueCUdeviceptr histogram_memory_;
  hololink::common::UniqueCUdeviceptr white_balance_gains_memory_;

  // Since this is in-place computation, we use an internal compute buffer to avoid 
  // HSDK-allocated separated input and output buffers.
  CUdeviceptr internal_compute_buffer_ = 0;

  uint32_t histogram_threadblock_size_ = 0;
};

}  // namespace hololink::operators

#endif /* HOLOHUB_IMAGE_PROCESSOR_GPU_RESIDENT_IMAGE_PROCESSOR_GPU_RESIDENT */
