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

namespace {

/**
 * Splits a string into a pair of strings, separated by a separator.
 *
 * @param s The string to split
 * @param separator The separator to use
 * @return A pair of strings, the first is the part before the colon, the second is the part after
 */
std::pair<std::string, std::string> split(const std::string& s, char separator) {
  auto colon_pos = s.find(separator);
  if (colon_pos != std::string::npos) {
    return {s.substr(0, colon_pos), s.substr(colon_pos + 1)};
  } else {
    return {s, ""};
  }
}

/**
 * Converts a shape to a uint3. The shape is expected to have the least varying dimension first, for
 * example an grayscale image with a width of 1024 and a height of 768 has a shape of (768, 1024,
 * 1).
 *
 * @param shape The shape to convert
 * @param skip_element_count If true, the element count is not included in the shape
 * @return A uint3
 */
uint3 shape_to_uint3(const nvidia::gxf::Shape& shape, bool skip_element_count) {
  uint3 result = make_uint3(1, 1, 1);
  uint32_t offset = skip_element_count ? 1 : 0;
  if (shape.rank() > 0 + offset) {
    result.x = shape.dimension(shape.rank() - 1 - offset);
    if (shape.rank() > 1 + offset) {
      result.y = shape.dimension(shape.rank() - 2 - offset);
      if (shape.rank() > 2 + offset) {
        result.z = shape.dimension(shape.rank() - 3 - offset);
        if (shape.rank() > 3 + offset) {
          throw std::runtime_error(
              fmt::format("Only tensors of rank <= {} are supported, found rank '{}'.",
                          3 + offset,
                          shape.rank()));
        }
      }
    }
  }
  return result;
}

nvidia::gxf::Shape swizzle_shape(const nvidia::gxf::Shape& shape, const std::string& swizzle) {
  // If the swizzle is not empty, we need to create a new shape based on the swizzle
  if (!swizzle.empty()) {
    std::vector<int32_t> result;

    for (auto item : swizzle) {
      if (item == 'c') {
        if (shape.rank() < 1) {
          throw std::runtime_error(
              fmt::format("Attribute 'alloc_size_of': swizzle '{}' not supported, reference tensor "
                          "has rank < 1.",
                          swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 1));
      } else if (item == 'x') {
        if (shape.rank() < 2) {
          throw std::runtime_error(
              fmt::format("Attribute 'alloc_size_of': swizzle '{}' not supported, reference tensor "
                          "has rank < 2.",
                          swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 2));
      } else if (item == 'y') {
        if (shape.rank() < 3) {
          throw std::runtime_error(
              fmt::format("Attribute 'alloc_size_of': swizzle '{}' not supported, reference tensor "
                          "has rank < 3.",
                          swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 3));
      } else if (item == 'z') {
        if (shape.rank() < 4) {
          throw std::runtime_error(
              fmt::format("Attribute 'alloc_size_of': swizzle '{}' not supported, reference tensor "
                          "has rank < 4.",
                          swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 4));
      } else if (item == 'w') {
        if (shape.rank() < 5) {
          throw std::runtime_error(
              fmt::format("Attribute 'alloc_size_of': swizzle '{}' not supported, reference tensor "
                          "has rank < 5.",
                          swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 5));
      } else if ((item > '0') && (item <= '9')) {
        result.insert(result.begin(), item - '0');
      } else {
        throw std::runtime_error(
            fmt::format("Attribute 'alloc_size_of': swizzle '{}' not supported.", swizzle));
      }
    }

    return nvidia::gxf::Shape(result);
  }

  // Return the original shape if no swizzle is provided
  return shape;
}

}  // namespace

namespace holoscan::ops {

CommandWorkspace::CommandWorkspace(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context)
    : op_input_(op_input), op_output_(op_output), context_(context) {}

CommandInput::CommandInput(const std::string& port_name, const std::string& item_name,
                           const std::string& resource_name, size_t parameter_offset)
    : port_name_(port_name),
      item_name_(item_name),
      resource_name_(resource_name),
      parameter_offset_(parameter_offset) {}

void CommandInput::execute(CommandWorkspace& workspace) {
  // Receive the entity from the input port
  HOLOSCAN_LOG_DEBUG("CommandInput: {}{} {}",
                     port_name_,
                     item_name_.empty() ? "" : ":" + item_name_,
                     resource_name_);

  // Receive the entity from the input port, it maybe cached in the workspace if the received entity
  // is expected to contain several items.
  holoscan::expected<gxf::Entity, holoscan::RuntimeError> maybe_entity;
  if (workspace.entities_.find(port_name_) == workspace.entities_.end()) {
    maybe_entity = workspace.op_input_.receive<gxf::Entity>(port_name_.c_str());
    if (!maybe_entity.has_value()) {
      throw std::runtime_error(
          fmt::format("Failed to receive entity from input port {}.", port_name_));
    }
    workspace.entities_[port_name_] = maybe_entity.value();
  } else {
    maybe_entity = workspace.entities_[port_name_];
  }

  /// @todo Warn on non-contiguous memory

  // Get the pointer and size of the data
  void* pointer;
  size_t size;

  auto maybe_tensor =
      static_cast<nvidia::gxf::Entity&>(maybe_entity.value())
          .get<nvidia::gxf::Tensor>(item_name_.empty() ? nullptr : item_name_.c_str());
  if (maybe_tensor) {
    // The entity is a tensor
    auto tensor = maybe_tensor.value();
    pointer = tensor->pointer();
    size = tensor->size();
  } else {
    auto maybe_video_buffer =
        static_cast<nvidia::gxf::Entity&>(maybe_entity.value())
            .get<nvidia::gxf::VideoBuffer>(item_name_.empty() ? nullptr : item_name_.c_str());
    if (!maybe_video_buffer) {
      throw std::runtime_error(fmt::format(
          "Attribute 'input': input '{}' is neither a tensor nor a video buffer.", port_name_));
    }
    // The entity is a video buffer
    auto video_buffer = maybe_video_buffer.value();
    pointer = video_buffer->pointer();
    size = video_buffer->size();
  }

  // Add the pointer to the CUDA resource pointers
  workspace.cuda_resource_pointers_[resource_name_] = {pointer, size};

  // Copy the tensor pointer to the shader parameters array
  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(void*));
  memcpy(workspace.shader_parameters_.data() + parameter_offset_, &pointer, sizeof(void*));
}

CommandOutput::CommandOutput(const std::string& port_name, const std::string& item_name,
                             const std::string& resource_name)
    : port_name_(port_name), item_name_(item_name), resource_name_(resource_name) {}

void CommandOutput::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandOutput: {}{} {}",
                     port_name_,
                     item_name_.empty() ? "" : ":" + item_name_,
                     resource_name_);
  // Make sure to emit the entity only once (we might have several items in the entity)
  if (workspace.emitted_entities_.find(port_name_) == workspace.emitted_entities_.end()) {
    workspace.op_output_.emit(workspace.entities_[port_name_], port_name_.c_str());
    workspace.emitted_entities_.insert(port_name_);
  }
}

CommandAllocSizeOf::CommandAllocSizeOf(const std::string& port_name, const std::string& item_name,
                                       const std::string& resource_name,
                                       const std::string& reference_name,
                                       const std::string& element_type, uint32_t element_count,
                                       const Parameter<std::shared_ptr<Allocator>>& allocator,
                                       size_t parameter_offset)
    : port_name_(port_name),
      item_name_(item_name),
      resource_name_(resource_name),
      reference_name_(reference_name),
      element_type_(element_type),
      element_count_(element_count),
      allocator_(allocator),
      parameter_offset_(parameter_offset) {}

void CommandAllocSizeOf::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandAllocSizeOf: {}{} {} {} {}{}",
                     port_name_,
                     item_name_.empty() ? "" : ":" + item_name_,
                     resource_name_,
                     reference_name_,
                     element_type_,
                     element_count_);

  auto [reference_remainder, swizzle] = split(reference_name_, '.');
  auto [reference_port_name, reference_item_name] = split(reference_remainder, ':');

  auto reference_entity = workspace.entities_.find(reference_port_name);
  if (reference_entity == workspace.entities_.end()) {
    throw std::runtime_error(
        fmt::format("Attribute 'alloc_size_of': input '{}' not found.", reference_port_name));
  }
  auto maybe_reference_tensor =
      static_cast<nvidia::gxf::Entity&>(reference_entity->second)
          .get<nvidia::gxf::Tensor>(reference_item_name.empty() ? nullptr
                                                                : reference_item_name.c_str());
  if (!maybe_reference_tensor) {
    throw std::runtime_error(
        fmt::format("Attribute 'alloc_size_of': input '{}' is not a tensor.", reference_name_));
  }
  auto reference_tensor = maybe_reference_tensor.value();

  // Create the entity if it does not exist
  holoscan::expected<gxf::Entity, holoscan::RuntimeError> maybe_entity;
  if (workspace.entities_.find(port_name_) == workspace.entities_.end()) {
    maybe_entity = gxf::Entity::New(&workspace.context_);
    workspace.entities_[port_name_] = maybe_entity.value();
  } else {
    maybe_entity = workspace.entities_[port_name_];
  }

  // Add a tensor to the entity
  auto tensor = static_cast<nvidia::gxf::Entity&>(maybe_entity.value())
                    .add<nvidia::gxf::Tensor>(item_name_.empty() ? nullptr : item_name_.c_str())
                    .value();

  // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<Allocator>
  auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      workspace.context_.context(), allocator_.get()->gxf_cid());

  std::vector<int32_t> result;
  if (!swizzle.empty()) {
    // Apply the swizzle to the reference tensor shape
    const nvidia::gxf::Shape swizzled_shape = swizzle_shape(reference_tensor->shape(), swizzle);
    for (int i = 0; i < swizzled_shape.rank(); ++i) {
      result.push_back(swizzled_shape.dimension(i));
    }
  } else {
    // Copy anything but the last dimension which is the element count
    const auto& shape = reference_tensor->shape();
    for (int i = 0; i < shape.rank() - 1; ++i) { result.push_back(shape.dimension(i)); }
  }
  // Replace the element count of the reference tensor with the element count of the shader resource
  result.push_back(element_count_);

  const nvidia::gxf::Shape alloc_shape = nvidia::gxf::Shape(result);

  // Reshape the tensor to the size of the reference tensor
  if (element_type_ == "int8") {
    tensor->reshape<int8_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "uint8") {
    tensor->reshape<uint8_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "int16") {
    tensor->reshape<int16_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "uint16") {
    tensor->reshape<uint16_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "int32") {
    tensor->reshape<int32_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "uint32") {
    tensor->reshape<uint32_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "int64") {
    tensor->reshape<int64_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "uint64") {
    tensor->reshape<uint64_t>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "float32") {
    tensor->reshape<float>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else if (element_type_ == "float64") {
    tensor->reshape<double>(alloc_shape, reference_tensor->storage_type(), gxf_allocator.value());
  } else {
    throw std::runtime_error(
        fmt::format("Attribute 'alloc_size_of': element type '{}' not supported.", element_type_));
  }

  void* const pointer = tensor->pointer();

  // Add the pointer to the CUDA resource pointers
  workspace.cuda_resource_pointers_[resource_name_] = {pointer, tensor->size()};

  // Copy the tensor pointer to the shader parameters array
  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(void*));
  memcpy(workspace.shader_parameters_.data() + parameter_offset_, &pointer, sizeof(void*));
}

void CommandParameter::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandParameter: {}", name_);
  workspace.shader_parameters_.resize(parameter_offset_ + size_);
  void* value_pointer = get_value_(value_.get());
  memcpy(workspace.shader_parameters_.data() + parameter_offset_, value_pointer, size_);
}

CommandSizeOf::CommandSizeOf(const std::string& parameter_name,
                             const std::string& reference_port_name, size_t parameter_offset)
    : parameter_name_(parameter_name),
      reference_port_name_(reference_port_name),
      parameter_offset_(parameter_offset) {}

void CommandSizeOf::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandSizeOf: {} {}", parameter_name_, reference_port_name_);

  auto [reference_port_name, reference_item_name] = split(reference_port_name_, ':');

  auto reference_entity = workspace.entities_.find(reference_port_name);
  if (reference_entity == workspace.entities_.end()) {
    throw std::runtime_error(
        fmt::format("Attribute 'size_of': input '{}' not found.", reference_port_name_));
  }
  auto maybe_reference_tensor =
      static_cast<nvidia::gxf::Entity&>(reference_entity->second)
          .get<nvidia::gxf::Tensor>(reference_item_name.empty() ? nullptr
                                                                : reference_item_name.c_str());
  if (!maybe_reference_tensor) {
    throw std::runtime_error(
        fmt::format("Attribute 'size_of': input '{}' is not a tensor.", reference_port_name_));
  }

  // Get the shape of the reference tensor and copy it to the shader parameters array
  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(dim3));
  uint3* param_data =
      reinterpret_cast<uint3*>(workspace.shader_parameters_.data() + parameter_offset_);
  *param_data = shape_to_uint3(maybe_reference_tensor.value()->shape(), true);
}

void CommandReceiveCudaStream::execute(CommandWorkspace& workspace) {
  workspace.cuda_stream_ = workspace.op_input_.receive_cuda_stream();
}

CommandLaunch::CommandLaunch(const std::string& name, SlangShader* shader, dim3 block_size,
                             const std::string& invocations_size_of_name, dim3 invocations)
    : name_(name),
      shader_(shader),
      block_size_(block_size),
      invocations_size_of_name_(invocations_size_of_name),
      invocations_(invocations) {
  // Get the kernel for the entry point
  kernel_ = shader_->get_kernel(name_);

  /// @todo Autodetect the optimal block size
  // CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&grid_size_, &block_size_, kernel_));
}

void CommandLaunch::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandLaunch: {}", name_);

  // Setup the invocation size
  dim3 invocations;
  if (!invocations_size_of_name_.empty()) {
    auto [invocations_size_of_remainder, swizzle] = split(invocations_size_of_name_, '.');
    auto [invocations_size_of_port_name, invocations_size_of_item_name] =
        split(invocations_size_of_remainder, ':');

    auto reference_entity = workspace.entities_.find(invocations_size_of_port_name);
    if (reference_entity == workspace.entities_.end()) {
      throw std::runtime_error(fmt::format("Attribute 'invocations_size_of': input '{}' not found.",
                                           invocations_size_of_name_));
    }
    auto maybe_reference_tensor =
        static_cast<nvidia::gxf::Entity&>(reference_entity->second)
            .get<nvidia::gxf::Tensor>(invocations_size_of_item_name.empty()
                                          ? nullptr
                                          : invocations_size_of_item_name.c_str());
    if (!maybe_reference_tensor) {
      throw std::runtime_error(
          fmt::format("Attribute 'invocations_size_of': input '{}' is not a tensor.",
                      invocations_size_of_name_));
    }

    const auto swizzled_shape = swizzle_shape(maybe_reference_tensor.value()->shape(), swizzle);
    // If we don't swizzle, we need to skip the element count, else the user is responsible for
    // setting the swizzle correctly
    invocations = shape_to_uint3(swizzled_shape, swizzle.empty());
  } else {
    invocations = invocations_;
  }

  const dim3 grid_size = dim3((invocations.x + block_size_.x - 1) / block_size_.x,
                              (invocations.y + block_size_.y - 1) / block_size_.y,
                              (invocations.z + block_size_.z - 1) / block_size_.z);

  // Set the global params
  shader_->update_global_params(name_, workspace.shader_parameters_, workspace.cuda_stream_);

  // Launch the kernel
  CUDA_CALL(cudaLaunchKernel(kernel_, grid_size, block_size_, nullptr, 0, workspace.cuda_stream_));
}

CommandZeros::CommandZeros(const std::string& resource_name) : resource_name_(resource_name) {}

void CommandZeros::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandZeros: {}", resource_name_);

  // Get the pointer and size of the data
  auto resource_info = workspace.cuda_resource_pointers_[resource_name_];

  // Initialize the data to zero
  CUDA_CALL(cudaMemsetAsync(resource_info.pointer, 0, resource_info.size, workspace.cuda_stream_));
}

}  // namespace holoscan::ops
