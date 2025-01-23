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

#include "tracking_postprocessor.h"
#include "stereo_depth_kernels.h"

namespace holoscan::ops {

void TrackingPostprocessor::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("yolo_input");
  spec.input<holoscan::gxf::Entity>("image");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(score_threshold_, "score_threshold", "score_threshold", "score_threshold", 0.0f);
  spec.param(bb_width_, "bb_width_", "bb_width_", "bb_width_", 640);
  spec.param(bb_height_, "bb_height_", "bb_height_", "bb_height_", 640);
}

void TrackingPostprocessor::compute(InputContext& op_input, OutputContext& op_output,
                                    ExecutionContext& context) {
  auto yolo_message = op_input.receive<gxf::Entity>("yolo_input").value();

  auto maybe_tensor = yolo_message.get<Tensor>("detection_boxes");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'detection_boxes' not found in message."); }
  auto detection_boxes = maybe_tensor;

  int boxes_batch = detection_boxes->shape()[0];
  int boxes_N = detection_boxes->shape()[1];
  int boxes_coord = detection_boxes->shape()[2];

  maybe_tensor = yolo_message.get<Tensor>("detection_scores");
  if (!maybe_tensor) {
    throw std::runtime_error("Tensor 'detection_scores' not found in message.");
  }
  auto detection_scores = maybe_tensor;

  maybe_tensor = yolo_message.get<Tensor>("num_detections");
  if (!maybe_tensor) { throw std::runtime_error("Tensor 'num_detections' not found in message."); }
  auto num_detections = maybe_tensor;

  int scores_batch = detection_boxes->shape()[0];
  int scores_N = detection_boxes->shape()[1];

  if (!(boxes_batch == 1 && scores_batch == 1 && boxes_N == scores_N && boxes_coord == 4)) {
    throw std::runtime_error("Wrong input dimensions for yolo input");
  }

  auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("image");
  const auto tensormap = maybe_tensormap.value();

  auto tensor = tensormap.begin()->second;
  int height = tensor->shape()[0];
  int width = tensor->shape()[1];
  int nChannels = tensor->shape()[2];

  if (nChannels != 3) {
    std::cout << "height " << height << "width " << width << "nChannels " << nChannels << std::endl;
    throw std::runtime_error("Expecting RGB image");
  }

  auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) { cudaFree(*pointer); }
      delete pointer;
    }
  });

  cudaMalloc(pointer.get(), width * height * 3 * sizeof(uint8_t));
  cudaMemcpy(*pointer,
             static_cast<void*>(tensor->data()),
             width * height * 3 * sizeof(uint8_t),
             cudaMemcpyDeviceToDevice);
  drawBB(static_cast<uint8_t*>(*pointer),
         static_cast<int32_t*>(num_detections->data()),
         static_cast<float*>(detection_boxes->data()),
         static_cast<float*>(detection_scores->data()),
         score_threshold_,
         width,
         height,
         bb_width_,
         bb_height_,
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
