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

#ifndef HOLOHUB_DISPLAY_GPU_RESIDENT_DISPLAY_GPU_RESIDENT
#define HOLOHUB_DISPLAY_GPU_RESIDENT_DISPLAY_GPU_RESIDENT

#include <cuDisp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/core/parameter.hpp>

namespace holoscan::ops {

enum DisplayOpSurfaceFormat { kDisplayOpSurfaceFormatA8R8G8B8 = 0 };

class DisplayGpuResidentOp : public holoscan::GPUResidentOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DisplayGpuResidentOp, holoscan::GPUResidentOperator);

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override;

 private:
  void initialize_backend();
  void cleanup_cudisp();

  holoscan::Parameter<int32_t> width_;
  holoscan::Parameter<int32_t> height_;
  holoscan::Parameter<int32_t> out_channels_;
  holoscan::Parameter<int32_t> element_size_;
  holoscan::Parameter<int32_t> display_width_;
  holoscan::Parameter<int32_t> display_height_;
  holoscan::Parameter<int32_t> refresh_rate_;
  holoscan::Parameter<int32_t> surface_format_;
  holoscan::Parameter<bool> gsync_;
  holoscan::Parameter<bool> front_buffer_rendering_;

  void** display_ptr_location_ptr_ = nullptr;
  void* display_ptr0_ = nullptr;
  void* display_ptr1_ = nullptr;
  void* present_handle_ = nullptr;
  cudaStream_t display_init_stream_ = nullptr;

  cuDispSwapchain swapchain_ = nullptr;
  CUdeviceptr display_device_ptr0_ = 0;
  CUdeviceptr display_device_ptr1_ = 0;
  void* display_ptrs_device_ = nullptr;
  void* display_ptr_locations_device_ = nullptr;
  unsigned int* num_buffers_per_layer_device_ = nullptr;
  uint64_t buffer_size_ = 0;

  unsigned short* resize_buffer_ = nullptr;
  uint32_t display_stride_ = 0;
  bool initialized_ = false;

  static constexpr unsigned int NUM_BUFFERS = 2;
};

}  // namespace holoscan::ops

#endif /* HOLOHUB_DISPLAY_GPU_RESIDENT_DISPLAY_GPU_RESIDENT */
