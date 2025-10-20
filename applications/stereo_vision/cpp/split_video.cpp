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

#include "split_video.h"
#include <npp.h>
#include <gxf/std/tensor.hpp>

namespace holoscan::ops {

void SplitVideoOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output1");
  spec.output<holoscan::gxf::Entity>("output2");
  spec.param(stereo_video_layout_,
             "stereo_video_layout",
             "Stereo Video Layout",
             "Horizontal or Vertical Concatenation of Stereo Video Frames",
             STEREO_VIDEO_HORIZONTAL);
}

void SplitVideoOp::compute(InputContext& op_input, OutputContext& op_output,
                           ExecutionContext& context) {
  auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("input");
  const auto tensormap = maybe_tensormap.value();

  if (tensormap.size() != 1) { throw std::runtime_error("Expecting single tensor input"); }

  auto tensor = tensormap.begin()->second;
  int height = tensor->shape()[0];
  int width = tensor->shape()[1];
  int nChannels = tensor->shape()[2];

  nvidia::gxf::Tensor tensor_gxf(tensor->dl_ctx());
  nvidia::gxf::PrimitiveType data_type = tensor_gxf.element_type();
  int element_size = nvidia::gxf::PrimitiveTypeSize(data_type);
  auto pointer1 = std::shared_ptr<void*>(new void*, [](void** pointer1) {
    if (pointer1 != nullptr) {
      if (*pointer1 != nullptr) { cudaFree(*pointer1); }
      delete pointer1;
    }
  });

  auto pointer2 = std::shared_ptr<void*>(new void*, [](void** pointer2) {
    if (pointer2 != nullptr) {
      if (*pointer2 != nullptr) { cudaFree(*pointer2); }
      delete pointer2;
    }
  });

  cudaMalloc(pointer1.get(), width * height * element_size * nChannels / 2);
  cudaMalloc(pointer2.get(), width * height * element_size * nChannels / 2);

  nvidia::gxf::Shape shape;
  if (stereo_video_layout_.get() == STEREO_VIDEO_VERTICAL) {
    cudaMemcpy(*pointer1,
               static_cast<void*>(tensor->data()),
               width * (height / 2) * element_size * nChannels,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        *pointer2,
        static_cast<void*>((char*)tensor->data() + width * (height / 2) * element_size * nChannels),
        width * (height / 2) * element_size * nChannels,
        cudaMemcpyDeviceToDevice);
    shape = nvidia::gxf::Shape{height / 2, width, nChannels};
  } else if (stereo_video_layout_.get() == STEREO_VIDEO_HORIZONTAL) {
    cudaMemcpy2D(*pointer1,
                 (width / 2) * element_size * nChannels,
                 static_cast<void*>(tensor->data()),
                 width * element_size * nChannels,
                 (width / 2) * element_size * nChannels,
                 height,
                 cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(*pointer2,
                 (width / 2) * element_size * nChannels,
                 static_cast<void*>((char*)tensor->data() + (width / 2) * element_size * nChannels),
                 width * element_size * nChannels,
                 (width / 2) * element_size * nChannels,
                 height,
                 cudaMemcpyDeviceToDevice);
    shape = nvidia::gxf::Shape{height, width / 2, nChannels};
  } else {
    throw std::runtime_error("UNKNOWN OUTPUT FORMAT");
  }

  auto out_message1 = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor1 = out_message1.value().add<nvidia::gxf::Tensor>("");

  gxf_tensor1.value()->wrapMemory(shape,
                                  data_type,
                                  element_size,
                                  nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                                  nvidia::gxf::MemoryStorageType::kDevice,
                                  *pointer1,
                                  [orig_pointer = pointer1](void*) mutable {
                                    orig_pointer.reset();  // decrement ref count
                                    return nvidia::gxf::Success;
                                  });

  op_output.emit(out_message1.value(), "output1");

  auto out_message2 = nvidia::gxf::Entity::New(context.context());
  auto gxf_tensor2 = out_message2.value().add<nvidia::gxf::Tensor>("");
  gxf_tensor2.value()->wrapMemory(shape,
                                  data_type,
                                  element_size,
                                  nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                                  nvidia::gxf::MemoryStorageType::kDevice,
                                  *pointer2,
                                  [orig_pointer = pointer2](void*) mutable {
                                    orig_pointer.reset();  // decrement ref count

                                    return nvidia::gxf::Success;
                                  });

  op_output.emit(out_message2.value(), "output2");
}

void MergeVideoOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input1");
  spec.input<holoscan::gxf::Entity>("input2");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(stereo_video_layout_,
             "stereo_video_layout",
             "Stereo Video Layout",
             "Horizontal or Vertical Concatenation of Stereo Video Frames",
             STEREO_VIDEO_HORIZONTAL);
}

void MergeVideoOp::compute(InputContext& op_input, OutputContext& op_output,
                           ExecutionContext& context) {
  auto maybe_tensormap1 = op_input.receive<holoscan::TensorMap>("input1");
  const auto tensormap1 = maybe_tensormap1.value();

  auto maybe_tensormap2 = op_input.receive<holoscan::TensorMap>("input2");
  const auto tensormap2 = maybe_tensormap2.value();

  if ((tensormap1.size() != 1) || (tensormap2.size() != 1)) {
    throw std::runtime_error("Expecting single tensor input");
  }

  auto tensor1 = tensormap1.begin()->second;
  auto tensor2 = tensormap2.begin()->second;

  int height = tensor1->shape()[0];
  int width = tensor1->shape()[1];
  int nChannels = tensor1->shape()[2];

  if ((height != tensor2->shape()[0]) || (width != tensor2->shape()[1]) ||
      (nChannels != tensor2->shape()[2])) {
    std::cout << "Tensor 1:" << height << "x" << width << "x" << nChannels << std::endl;
    std::cout << "Tensor 2:" << tensor2->shape()[0] << "x" << tensor2->shape()[1] << "x"
              << tensor2->shape()[2] << std::endl;

    throw std::runtime_error("IMAGE SIZES DO NOT MATCH");
  }

  nvidia::gxf::Tensor tensor_gxf(tensor1->dl_ctx());
  nvidia::gxf::PrimitiveType data_type = tensor_gxf.element_type();
  int element_size = nvidia::gxf::PrimitiveTypeSize(data_type);

  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) { cudaFree(*pointer); }
      delete pointer;
    }
  });
  cudaMalloc(pointer.get(), width * height * element_size * nChannels * 2);

  nvidia::gxf::Shape shape;
  if (stereo_video_layout_.get() == STEREO_VIDEO_VERTICAL) {
    cudaMemcpy(*pointer,
               static_cast<void*>(tensor1->data()),
               width * height * element_size * nChannels,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(static_cast<void*>(static_cast<char*>(*pointer) +
                                  width * height * element_size * nChannels),
               static_cast<void*>(tensor2->data()),
               width * height * element_size * nChannels,
               cudaMemcpyDeviceToDevice);
    shape = nvidia::gxf::Shape{height * 2, width, nChannels};
  } else if (stereo_video_layout_.get() == STEREO_VIDEO_HORIZONTAL) {
    cudaMemcpy2D(*pointer,
                 width * 2 * element_size * nChannels,
                 static_cast<void*>(tensor1->data()),
                 width * element_size * nChannels,
                 width * element_size * nChannels,
                 height,
                 cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(
        static_cast<void*>(static_cast<char*>(*pointer) + width * element_size * nChannels),
        width * 2 * element_size * nChannels,
        static_cast<void*>(tensor2->data()),
        width * element_size * nChannels,
        width * element_size * nChannels,
        height,
        cudaMemcpyDeviceToDevice);
    shape = nvidia::gxf::Shape{height, width * 2, nChannels};
  } else {
    throw std::runtime_error("UNKNOWN OUTPUT FORMAT");
  }

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
