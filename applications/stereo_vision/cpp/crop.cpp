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

#include "crop.h"
#include <gxf/std/tensor.hpp>

namespace holoscan::ops {

void CropOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(x_, "x", "top left x", "top left x coordinate", 0);
  spec.param(y_, "y", "top left y", "top left y coordinate", 0);
  spec.param(width_, "width", "width", "width", 0);
  spec.param(height_, "height", "height", "height", 0);
}

void CropOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
  auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("input");
  const auto tensormap = maybe_tensormap.value();

  if (tensormap.size() != 1) {
    throw std::runtime_error("Expecting single tensor input");
  }

  auto tensor = tensormap.begin()->second;
  int orig_height = tensor->shape()[0];
  int orig_width = tensor->shape()[1];
  int nChannels = tensor->shape()[2];

  nvidia::gxf::Tensor tensor_gxf(tensor->dl_ctx());
  nvidia::gxf::PrimitiveType data_type = tensor_gxf.element_type();
  int element_size = nvidia::gxf::PrimitiveTypeSize(data_type);

  if (x_ < 0 || y_ < 0 || width_ <= 0 || height_ <= 0) {
    throw std::runtime_error("Invalid crop dimensions");
  }

  if ((x_ + width_) > orig_width || (y_ + height_) > orig_height) {
    throw std::runtime_error("Crop exceeds image boundaries");
  }

  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) {
        cudaFree(*pointer);
      }
      delete pointer;
    }
  });
  cudaMalloc(pointer.get(), width_ * height_ * element_size * nChannels);

  nvidia::gxf::Shape shape = nvidia::gxf::Shape{height_, width_, nChannels};
  cudaMemcpy2D(*pointer,
               width_ * element_size * nChannels,
               static_cast<void*>((char*)tensor->data() + x_ * element_size * nChannels),
               orig_width * element_size * nChannels,
               width_ * element_size * nChannels,
               height_,
               cudaMemcpyDeviceToDevice);

  auto out_message = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("");

  gxf_tensor.value()->wrapMemory(shape,
                                 data_type,
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
