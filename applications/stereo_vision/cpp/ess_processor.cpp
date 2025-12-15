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

#include "ess_processor.h"
#include <npp.h>
#include "stereo_depth_kernels.h"

namespace holoscan::ops {

void ESSPreprocessorOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input1");
  spec.input<holoscan::gxf::Entity>("input2");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(width_, "width", "width", "width", 0);
  spec.param(height_, "height", "height", "height", 0);
}

void ESSPreprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  auto maybe_tensormap1 = op_input.receive<holoscan::TensorMap>("input1");
  const auto tensormap1 = maybe_tensormap1.value();

  auto maybe_tensormap2 = op_input.receive<holoscan::TensorMap>("input2");
  const auto tensormap2 = maybe_tensormap2.value();

  auto tensor1 = tensormap1.begin()->second;
  auto tensor2 = tensormap2.begin()->second;

  int orig_height = tensor1->shape()[0];
  int orig_width = tensor1->shape()[1];
  int nChannels = tensor1->shape()[2];

  if ((orig_height != tensor2->shape()[0]) || (orig_width != tensor2->shape()[1]) ||
      (nChannels != tensor2->shape()[2])) {
    throw std::runtime_error("IMAGE SIZES DO NOT MATCH");
  }

  if ((tensormap1.size() != 1) || (tensormap2.size() != 1)) {
    throw std::runtime_error("Expecting single tensor input");
  }

  if (!(nChannels == 3 || nChannels == 4)) {
    throw std::runtime_error("Input tensor must have 3 or 4 channels");
  }

  auto pointerLeft = std::shared_ptr<void*>(new void*, [](void** pointerLeft) {
    if (pointerLeft != nullptr) {
      if (*pointerLeft != nullptr) {
        cudaFree(*pointerLeft);
      }
      delete pointerLeft;
    }
  });

  auto pointerRight = std::shared_ptr<void*>(new void*, [](void** pointerRight) {
    if (pointerRight != nullptr) {
      if (*pointerRight != nullptr) {
        cudaFree(*pointerRight);
      }
      delete pointerRight;
    }
  });

  cudaMalloc(pointerLeft.get(), width_ * height_ * sizeof(float) * 3);
  cudaMalloc(pointerRight.get(), width_ * height_ * sizeof(float) * 3);

  nvidia::gxf::Shape shape = nvidia::gxf::Shape{1, 3, height_, width_};
  preprocessESS(static_cast<uint8_t*>(tensor1->data()),
                static_cast<float*>(*pointerLeft.get()),
                orig_width,
                orig_height,
                nChannels,
                width_,
                height_,
                0);
  preprocessESS(static_cast<uint8_t*>(tensor2->data()),
                static_cast<float*>(*pointerRight.get()),
                orig_width,
                orig_height,
                nChannels,
                width_,
                height_,
                0);

  auto out_message = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor_left = out_message.value().add<nvidia::gxf::Tensor>("input_left");

  gxf_tensor_left.value()->wrapMemory(shape,
                                      nvidia::gxf::PrimitiveType::kFloat32,
                                      sizeof(float),
                                      nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float)),
                                      nvidia::gxf::MemoryStorageType::kDevice,
                                      *pointerLeft,
                                      [orig_pointer = pointerLeft](void*) mutable {
                                        orig_pointer.reset();  // decrement ref count
                                        return nvidia::gxf::Success;
                                      });

  auto gxf_tensor_right = out_message.value().add<nvidia::gxf::Tensor>("input_right");

  gxf_tensor_right.value()->wrapMemory(shape,
                                       nvidia::gxf::PrimitiveType::kFloat32,
                                       sizeof(float),
                                       nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float)),
                                       nvidia::gxf::MemoryStorageType::kDevice,
                                       *pointerRight,
                                       [orig_pointer = pointerRight](void*) mutable {
                                         orig_pointer.reset();  // decrement ref count
                                         return nvidia::gxf::Success;
                                       });

  op_output.emit(out_message.value(), "output");
}

void ESSPostprocessorOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(width_, "width", "width", "width", 0);
  spec.param(height_, "height", "height", "height", 0);
}

void ESSPostprocessorOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("input");
  const auto tensormap = maybe_tensormap.value();

  auto tensor_disp = tensormap.at("output_left");
  auto tensor_conf = tensormap.at("output_conf");

  int orig_height = tensor_disp->shape()[1];
  int orig_width = tensor_disp->shape()[2];

  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) {
        cudaFree(*pointer);
      }
      delete pointer;
    }
  });

  cudaMalloc(pointer.get(), width_ * height_ * sizeof(float));

  confidenceMask(
      (float*)tensor_disp->data(), (float*)tensor_conf->data(), 0.05, orig_width, orig_height, 0);

  NppStatus status = nppiResize_32f_C1R(static_cast<Npp32f*>(tensor_disp->data()),
                                        orig_width * sizeof(Npp32f),
                                        {orig_width, orig_height},
                                        {0, 0, orig_width, orig_height},
                                        (Npp32f*)*pointer,
                                        width_ * sizeof(Npp32f),
                                        {width_, height_},
                                        {0, 0, width_, height_},
                                        NPPI_INTER_NN);

  status = nppiMulC_32f_C1IR(static_cast<Npp32f>(width_) / static_cast<Npp32f>(orig_width),
                             (Npp32f*)*pointer,
                             width_ * sizeof(Npp32f),
                             {width_, height_});

  nvidia::gxf::Shape shape = nvidia::gxf::Shape{height_, width_, 1};

  auto out_message = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("");

  gxf_tensor.value()->wrapMemory(shape,
                                 nvidia::gxf::PrimitiveType::kFloat32,
                                 sizeof(float),
                                 nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float)),
                                 nvidia::gxf::MemoryStorageType::kDevice,
                                 *pointer,
                                 [orig_pointer = pointer](void*) mutable {
                                   orig_pointer.reset();  // decrement ref count
                                   return nvidia::gxf::Success;
                                 });

  op_output.emit(out_message.value(), "output");
}

}  // namespace holoscan::ops
