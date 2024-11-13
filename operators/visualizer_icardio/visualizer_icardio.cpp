/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <utility>

#include "visualizer_icardio.hpp"

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/utils/holoinfer_utils.hpp"

#include <cuda_runtime.h>

#include "visualizer_icardio.cuh"

namespace holoscan::ops {

void VisualizerICardioOp::setup(OperatorSpec& spec) {
  auto& out_tensor_1 = spec.output<gxf::Entity>("keypoints");
  auto& out_tensor_2 = spec.output<gxf::Entity>("keyarea_1");
  auto& out_tensor_3 = spec.output<gxf::Entity>("keyarea_2");
  auto& out_tensor_4 = spec.output<gxf::Entity>("keyarea_3");
  auto& out_tensor_5 = spec.output<gxf::Entity>("keyarea_4");
  auto& out_tensor_6 = spec.output<gxf::Entity>("keyarea_5");
  auto& out_tensor_7 = spec.output<gxf::Entity>("lines");
  auto& out_tensor_8 = spec.output<gxf::Entity>("logo");

  spec.param(
      in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {std::string("")});
  spec.param(
      out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {std::string("")});
  spec.param(data_dir_,
             "data_dir",
             "Data Directory",
             "Data directory",
             {std::string("../data/multiai_ultrasound")});
  spec.param(input_on_cuda_, "input_on_cuda", "Input buffer on CUDA", "", true);
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  spec.param(receivers_, "receivers", "Receivers", "List of receivers", {});
  spec.param(transmitters_,
             "transmitters",
             "Transmitters",
             "List of transmitters",
             {&out_tensor_1,
              &out_tensor_2,
              &out_tensor_3,
              &out_tensor_4,
              &out_tensor_5,
              &out_tensor_6,
              &out_tensor_7,
              &out_tensor_8});
  cuda_stream_handler_.define_params(spec);
}

void VisualizerICardioOp::start() {
  std::vector<int> logo_dim = tensor_to_shape_.at("logo");
  size_t logo_size =
      std::accumulate(logo_dim.begin(), logo_dim.end(), 1, std::multiplies<size_t>());

  logo_image_ = allocator_.get()->allocate(logo_size, holoscan::MemoryStorageType::kDevice);
  if (!logo_image_) { HoloInfer::raise_error(module_, "Failed to allocate logo memory."); }

  std::string path_to_logo_file = data_dir_.get() + "/" + logo_file_;
  std::ifstream file_logo(path_to_logo_file);

  if (!file_logo) {
    HOLOSCAN_LOG_WARN("Logo file " + path_to_logo_file + " not found, Ignored.");
    CUDA_TRY(cudaMemset(logo_image_, 0, logo_size));
  } else {
    std::istream_iterator<int> start(file_logo), end;
    std::vector<int> data_logo(start, end);

    if (data_logo.size() != logo_size) {
      HoloInfer::raise_error(module_, "Logo file size mismatch.");
    }

    std::vector<uint8_t> image_data;
    std::copy(std::begin(data_logo), std::end(data_logo), std::back_inserter(image_data));

    CUDA_TRY(cudaMemcpy(logo_image_, image_data.data(), logo_size, cudaMemcpyHostToDevice));
  }
}

void VisualizerICardioOp::stop() {
  if (logo_image_) {
    allocator_.get()->free(reinterpret_cast<nvidia::byte*>(logo_image_));
    logo_image_ = nullptr;
  }
}

void VisualizerICardioOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context) {
  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto cid = allocator_.get()->gxf_cid();
  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), cid);
  auto cont = context.context();

  try {
    HoloInfer::DataMap data_per_tensor;
#if HOLOSCAN_MAJOR_VERSION == 0 && HOLOSCAN_MINOR_VERSION < 6
    gxf_result_t stat = holoscan::utils::multiai_get_data_per_model(op_input,
                                                                    in_tensor_names_.get(),
                                                                    data_per_tensor,
                                                                    tensor_size_map_,
                                                                    input_on_cuda_.get(),
                                                                    module_);
#else
    gxf_result_t stat = holoscan::utils::get_data_per_model(op_input,
                                                            in_tensor_names_.get(),
                                                            data_per_tensor,
                                                            tensor_size_map_,
                                                            input_on_cuda_.get(),
                                                            module_,
                                                            cont,
                                                            cuda_stream_handler_);
#endif
    if (stat != GXF_SUCCESS) { HoloInfer::raise_error(module_, "Tick, Data extraction"); }

    if (tensor_size_map_.find(pc_tensor_name_) == tensor_size_map_.end()) {
      HoloInfer::report_error(module_, "Dimension not found for tensor " + pc_tensor_name_);
    }
    if (data_per_tensor.find(pc_tensor_name_) == data_per_tensor.end()) {
      HoloInfer::report_error(module_, "Data not found for tensor " + pc_tensor_name_);
    }
    const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(cont);
    auto coords = static_cast<float*>(data_per_tensor.at(pc_tensor_name_)->device_buffer->data());
    auto datasize = tensor_size_map_[pc_tensor_name_];

    if (transmitters_.get().size() > 0) {
      for (unsigned int a = 0; a < transmitters_.get().size(); ++a) {
        auto out_message = nvidia::gxf::Entity::New(context.context());
        if (!out_message) { HoloInfer::raise_error(module_, "Tick, Out message allocation"); }
        std::string current_tensor_name{out_tensor_names_.get()[a]};
        auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(current_tensor_name.c_str());
        if (!out_tensor) { HoloInfer::raise_error(module_, "Tick, Out tensor allocation"); }
        if (tensor_to_shape_.find(current_tensor_name) == tensor_to_shape_.end()) {
          HoloInfer::raise_error(
              module_, "Tick, Output Tensor shape mapping not found for " + current_tensor_name);
        }
        std::vector<int> shape_dim = tensor_to_shape_.at(current_tensor_name);
        nvidia::gxf::Shape output_shape{shape_dim[0], shape_dim[1], shape_dim[2]};

        if (current_tensor_name.compare("logo") == 0) {
          nvidia::gxf::Shape shape{shape_dim[0], shape_dim[1], shape_dim[2]};
          auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
          auto bytes_per_element = nvidia::gxf::PrimitiveTypeSize(type);
          auto strides = nvidia::gxf::ComputeTrivialStrides(shape, bytes_per_element);
          if (!out_tensor.value()->wrapMemory(shape,
                                              type,
                                              bytes_per_element,
                                              strides,
                                              nvidia::gxf::MemoryStorageType::kDevice,
                                              logo_image_,
                                              nullptr)) {
            HoloInfer::raise_error(module_, "Tick, wrap logo memory.");
          }
        } else {
          out_tensor.value()->reshape<float>(
              output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
          if (!out_tensor.value()->pointer()) {
            HoloInfer::raise_error(module_, "Tick, Out tensor buffer allocation");
          }
          nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
          if (!out_tensor_data) {
            HoloInfer::raise_error(module_, "Tick, Getting out tensor data");
          }
          float* out_tensor_buffer = out_tensor_data.value();

          int property_size = shape_dim[2];

          if (property_size <= 3) {
            gen_coords(1,
                       (datasize[datasize.size() - 1] / 2) - 1,
                       property_size,
                       coords,
                       out_tensor_buffer,
                       cuda_stream);
          } else {
            if (tensor_to_index_.find(current_tensor_name) == tensor_to_index_.end()) {
              HoloInfer::raise_error(module_, "Tick, tensor to index mapping failed");
            }
            int index_coord = tensor_to_index_.at(current_tensor_name);
            if (index_coord >= 1 && index_coord <= 5) {
              gen_coords(index_coord, 1, property_size, coords, out_tensor_buffer, cuda_stream);
            } else {
              HoloInfer::raise_error(module_, "Tick, invalid coordinate from tensor");
            }
          }
        }
        cuda_stream_handler_.to_message(out_message);
        auto result = gxf::Entity(std::move(out_message.value()));
        op_output.emit(result, current_tensor_name.c_str());
      }
    }
  } catch (const std::runtime_error& r_) {
    HoloInfer::raise_error(module_, "Tick, Message->" + std::string(r_.what()));
  } catch (...) { HoloInfer::raise_error(module_, "Tick, unknown exception"); }
}

}  // namespace holoscan::ops
