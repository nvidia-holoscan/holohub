/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "heat_map.h"
#include "stereo_depth_kernels.h"

namespace holoscan::ops {

void HeatmapOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(min_disp_, "min_disp", "min_disp", "min_disp", 0.0f);
  spec.param(max_disp_, "max_disp", "max_disp", "max_disp", 255.0f);
}

void HeatmapOp::compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext& context) {
  auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("input");
  const auto tensormap = maybe_tensormap.value();

  if (tensormap.size() != 1) { throw std::runtime_error("Expecting single tensor input"); }

  auto tensor = tensormap.begin()->second;
  int height = tensor->shape()[0];
  int width = tensor->shape()[1];
  int nChannels = tensor->shape()[2];

  if (nChannels != 1) {
    std::cout << "height " << height << "width " << width << "nChannels " << nChannels << std::endl;
    throw std::runtime_error("Expecting grayscale input");
  }

  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) { cudaFree(*pointer); }
      delete pointer;
    }
  });

  cudaMalloc(pointer.get(), width * height * 3 * sizeof(uint8_t));
  heatmapF32(static_cast<float*>(tensor->data()),
             static_cast<uint8_t*>(*pointer),
             min_disp_,
             max_disp_,
             width,
             height,
             0);
  auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
  int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
  nvidia::gxf::Shape shape = nvidia::gxf::Shape{height, width, 3};

  auto out_message = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("");

  gxf_tensor.value()->wrapMemory(shape,
                                 nvidia::gxf::PrimitiveType::kUnsigned8,
                                 element_size,
                                 nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                                 nvidia::gxf::MemoryStorageType::kDevice,
                                 *pointer,
                                 [orig_pointer = pointer](void*) mutable {
                                   orig_pointer.reset();  // decrement ref count
                                   return nvidia::gxf::Success;
                                 });

  op_output.emit(out_message.value(), "output");
}

}  // namespace holoscan::ops
