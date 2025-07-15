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

#include "command.hpp"

#include "slang_shader.hpp"

namespace holoscan::ops {

CommandWorkspace::CommandWorkspace(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context)
    : op_input_(op_input), op_output_(op_output), context_(context) {}

CommandInput::CommandInput(const std::string& port_name, const std::string& resource_name,
                           size_t parameter_offset)
    : port_name_(port_name), resource_name_(resource_name), parameter_offset_(parameter_offset) {}

void CommandInput::execute(CommandWorkspace& workspace) {
  // Receive the entity from the input port
  HOLOSCAN_LOG_DEBUG("CommandInput: {} {}", port_name_, resource_name_);
  auto maybe_entity = workspace.op_input_.receive<gxf::Entity>(port_name_.c_str());
  if (!maybe_entity.has_value()) {
    throw std::runtime_error(
        fmt::format("Failed to receive entity from input port {}.", port_name_));
  }
  workspace.entities_[resource_name_] = maybe_entity.value();

  // Get the pointer to the data
  void* pointer = nullptr;

  auto maybe_tensor =
      static_cast<nvidia::gxf::Entity&>(maybe_entity.value()).get<nvidia::gxf::Tensor>();
  if (maybe_tensor) {
    // The entity is a tensor
    auto tensor = maybe_tensor.value();
    pointer = tensor->pointer();
  } else {
    auto maybe_video_buffer =
        static_cast<nvidia::gxf::Entity&>(maybe_entity.value()).get<nvidia::gxf::VideoBuffer>();
    if (!maybe_video_buffer) {
      throw std::runtime_error(fmt::format(
          "Attribute 'input': input '{}' is neither a tensor nor a video buffer.", port_name_));
    }
    // The entity is a video buffer
    auto video_buffer = maybe_video_buffer.value();
    pointer = video_buffer->pointer();
  }

  // Copy the tensor pointer to the shader parameters array
  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(void*));
  memcpy(workspace.shader_parameters_.data() + parameter_offset_, &pointer, sizeof(void*));
}

CommandOutput::CommandOutput(const std::string& port_name, const std::string& resource_name)
    : port_name_(port_name), resource_name_(resource_name) {}

void CommandOutput::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandOutput: {} {}", port_name_, resource_name_);
  workspace.op_output_.emit(workspace.entities_[resource_name_], port_name_.c_str());
}

CommandAllocSizeOf::CommandAllocSizeOf(const std::string& name, const std::string& size_of_name,
                                       const Parameter<std::shared_ptr<Allocator>>& allocator,
                                       size_t parameter_offset)
    : name_(name),
      size_of_name_(size_of_name),
      allocator_(allocator),
      parameter_offset_(parameter_offset) {}

void CommandAllocSizeOf::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandAllocSizeOf: {} {}", name_, size_of_name_);
  auto reference_entity = workspace.entities_.find(size_of_name_);
  if (reference_entity == workspace.entities_.end()) {
    throw std::runtime_error(
        fmt::format("Attribute 'size_of': input '{}' not found.", size_of_name_));
  }
  auto maybe_reference_tensor =
      static_cast<nvidia::gxf::Entity&>(reference_entity->second).get<nvidia::gxf::Tensor>();
  if (!maybe_reference_tensor) {
    throw std::runtime_error(
        fmt::format("Attribute 'size_of': input '{}' is not a tensor.", size_of_name_));
  }
  auto reference_tensor = maybe_reference_tensor.value();

  auto entity = gxf::Entity::New(&workspace.context_);
  auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>().value();
  // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<Allocator>
  auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      workspace.context_.context(), allocator_.get()->gxf_cid());
  tensor->reshape<int>(
      reference_tensor->shape(), nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());

  workspace.entities_[name_] = entity;

  // Copy the tensor pointer to the shader parameters array
  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(void*));
  void* pointer = tensor->pointer();
  memcpy(workspace.shader_parameters_.data() + parameter_offset_, &pointer, sizeof(void*));
}

void CommandParameter::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandParameter: {}", name_);
  workspace.shader_parameters_.resize(parameter_offset_ + size_);
  void* value_pointer = get_value_(value_.get());
  memcpy(workspace.shader_parameters_.data() + parameter_offset_, value_pointer, size_);
}

CommandSizeOf::CommandSizeOf(const std::string& name, const std::string& size_of_name,
                             size_t parameter_offset)
    : name_(name), size_of_name_(size_of_name), parameter_offset_(parameter_offset) {}

void CommandSizeOf::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandSizeOf: {} {}", name_, size_of_name_);
  auto reference_entity = workspace.entities_.find(size_of_name_);
  if (reference_entity == workspace.entities_.end()) {
    throw std::runtime_error(
        fmt::format("Attribute 'size_of': input '{}' not found.", size_of_name_));
  }
  auto maybe_reference_tensor =
      static_cast<nvidia::gxf::Entity&>(reference_entity->second).get<nvidia::gxf::Tensor>();
  if (!maybe_reference_tensor) {
    throw std::runtime_error(
        fmt::format("Attribute 'size_of': input '{}' is not a tensor.", size_of_name_));
  }
  auto reference_tensor = maybe_reference_tensor.value();

  if (reference_tensor->rank() > 3) {
    throw std::runtime_error(fmt::format("Only tensors of rank < 4 are supported, found rank '{}'.",
                                         reference_tensor->rank()));
  }

  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(int) * 3);

  // Get the shape of the reference tensor and copy it to the shader parameters array
  auto shape = reference_tensor->shape();

  if (reference_tensor->rank() > 3) {
    throw std::runtime_error(fmt::format("Only tensors of rank <4 are supported, found rank '{}'.",
                                         reference_tensor->rank()));
  }

  int32_t* param_data =
      reinterpret_cast<int32_t*>(workspace.shader_parameters_.data() + parameter_offset_);
  param_data[0] = shape.dimension(0);
  param_data[1] = shape.dimension(1);
  param_data[2] = shape.dimension(2);
}

void CommandReceiveCudaStream::execute(CommandWorkspace& workspace) {
  workspace.cuda_stream_ = workspace.op_input_.receive_cuda_stream();
}

CommandLaunch::CommandLaunch(const std::string& name, SlangShader* shader, dim3 block_size,
                             const std::string& grid_size_of_name, dim3 grid_size)
    : name_(name),
      shader_(shader),
      block_size_(block_size),
      grid_size_of_name_(grid_size_of_name),
      grid_size_(grid_size) {
  // Get the kernel for the entry point
  kernel_ = shader_->get_kernel(name_);
}

void CommandLaunch::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandLaunch: {}", name_);

  // Setup the grid size
  dim3 grid_size(1, 1, 1);
  if (!grid_size_of_name_.empty()) {
    auto reference_entity = workspace.entities_.find(grid_size_of_name_);
    if (reference_entity == workspace.entities_.end()) {
      throw std::runtime_error(
          fmt::format("Attribute 'grid_size_of': input '{}' not found.", grid_size_of_name_));
    }
    auto maybe_reference_tensor =
        static_cast<nvidia::gxf::Entity&>(reference_entity->second).get<nvidia::gxf::Tensor>();
    if (!maybe_reference_tensor) {
      throw std::runtime_error(
          fmt::format("Attribute 'grid_size_of': input '{}' is not a tensor.", grid_size_of_name_));
    }
    auto reference_tensor = maybe_reference_tensor.value();
    auto shape = reference_tensor->shape();
    grid_size.x = shape.dimension(0);
    if (shape.rank() > 1) {
      grid_size.y = shape.dimension(1);
      if (shape.rank() > 2) {
        grid_size.z = shape.dimension(2);
      }
    }
  } else {
    grid_size = grid_size_;
  }

  grid_size = dim3((grid_size.x + block_size_.x - 1) / block_size_.x,
                   (grid_size.y + block_size_.y - 1) / block_size_.y,
                   (grid_size.z + block_size_.z - 1) / block_size_.z);

  // Set the global params
  shader_->update_global_params(workspace.shader_parameters_);

  // Launch the kernel
  CUDA_CALL(cudaLaunchKernel(kernel_, grid_size, block_size_, nullptr, 0, workspace.cuda_stream_));
}

}  // namespace holoscan::ops
