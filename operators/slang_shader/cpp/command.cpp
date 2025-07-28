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

#include "slang_shader_compiler.hpp"

namespace {

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
  // Only skip the element count if this is not an one-dimensional tensor
  uint32_t offset = (skip_element_count && (shape.rank() > 1)) ? 1 : 0;
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

/**
 * Swizzles a shape based on a string. The string is a sequence of characters that specify the
 * dimensions of the shape. The characters are:
 * - 'c': the component count
 * - 'x': the first dimension
 * - 'y': the second dimension
 * - 'z': the third dimension
 * - 'w': the fourth dimension
 * - '0-9': the dimension number
 *
 * @param shape The shape to swizzle
 * @param swizzle The swizzle string
 * @return The swizzled shape
 */
nvidia::gxf::Shape swizzle_shape(const nvidia::gxf::Shape& shape, const std::string& swizzle) {
  // If the swizzle is not empty, we need to create a new shape based on the swizzle
  if (!swizzle.empty()) {
    std::vector<int32_t> result;

    for (auto item : swizzle) {
      if (item == 'c') {
        if (shape.rank() < 1) {
          throw std::runtime_error(
              fmt::format("Swizzle '{}' not supported, reference resource has rank < 1.", swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 1));
      } else if (item == 'x') {
        if (shape.rank() < 2) {
          throw std::runtime_error(
              fmt::format("Swizzle '{}' not supported, reference resource has rank < 2.", swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 2));
      } else if (item == 'y') {
        if (shape.rank() < 3) {
          throw std::runtime_error(
              fmt::format("Swizzle '{}' not supported, reference resource has rank < 3.", swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 3));
      } else if (item == 'z') {
        if (shape.rank() < 4) {
          throw std::runtime_error(
              fmt::format("Swizzle '{}' not supported, reference resource has rank < 4.", swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 4));
      } else if (item == 'w') {
        if (shape.rank() < 5) {
          throw std::runtime_error(
              fmt::format("Swizzle '{}' not supported, reference resource has rank < 5.", swizzle));
        }
        result.insert(result.begin(), shape.dimension(shape.rank() - 5));
      } else if ((item > '0') && (item <= '9')) {
        result.insert(result.begin(), item - '0');
      } else {
        throw std::runtime_error(fmt::format("Swizzle '{}' not supported.", swizzle));
      }
    }

    return nvidia::gxf::Shape(result);
  }

  // Return the original shape if no swizzle is provided
  return shape;
}

}  // namespace

namespace holoscan::ops {

/**
 * Retrieves the shape, strides, and storage type of a reference input from the workspace based on
 * the reference name.
 * The reference name can be in the format "port_name" or "port_name:item_name".
 *
 * @param workspace The command workspace containing entities
 * @param reference_name The name of the reference input (format: "port_name" or
 * "port_name:item_name")
 * @param attribute_name The name of the attribute for error reporting
 * @return A tuple containing the shape, strides, and storage type of the reference input
 * @throws std::runtime_error if the reference input is not found or is not a tensor or video buffer
 */
static std::tuple<nvidia::gxf::Shape, std::vector<uint64_t>, nvidia::gxf::MemoryStorageType>
get_reference_specs(CommandWorkspace& workspace, const std::string& reference_name,
                    const std::string& attribute_name) {
  nvidia::gxf::Shape shape;
  std::vector<uint64_t> strides;
  nvidia::gxf::MemoryStorageType storage_type;

  auto [reference_port_name, reference_item_name] = split(reference_name, ':');

  auto reference_port_info_it = workspace.port_info_.find(reference_port_name);
  if (reference_port_info_it == workspace.port_info_.end()) {
    throw std::runtime_error(
        fmt::format("Attribute '{}': input '{}' not found.", attribute_name, reference_name));
  }
  auto maybe_reference_tensor =
      static_cast<nvidia::gxf::Entity&>(reference_port_info_it->second->entity_)
          .get<nvidia::gxf::Tensor>(reference_item_name.empty() ? nullptr
                                                                : reference_item_name.c_str());
  if (maybe_reference_tensor) {
    shape = maybe_reference_tensor.value()->shape();
    strides.resize(shape.rank());
    for (uint32_t i = 0; i < shape.rank(); ++i) {
      strides[i] = maybe_reference_tensor.value()->stride(i);
    }
    storage_type = maybe_reference_tensor.value()->storage_type();
  } else {
    auto maybe_reference_video_buffer =
        static_cast<nvidia::gxf::Entity&>(reference_port_info_it->second->entity_)
            .get<nvidia::gxf::VideoBuffer>(
                reference_item_name.empty() ? nullptr : reference_item_name.c_str());
    if (maybe_reference_video_buffer) {
      const nvidia::gxf::VideoBufferInfo& video_frame_info =
          maybe_reference_video_buffer.value()->video_frame_info();
      if (video_frame_info.color_planes.size() != 1) {
        throw std::runtime_error(
            fmt::format("Attribute '{}': input '{}' VideoBuffer with multiple color planes is not "
                        "supported.",
                        attribute_name,
                        reference_name));
      }

      auto maybe_primitive_type =
          nvidia::gxf::VideoBuffer::getPlanarPrimitiveType(video_frame_info.color_format);
      if (!maybe_primitive_type) {
        throw std::runtime_error(
            fmt::format("Attribute '{}': input '{}' VideoBuffer with invalid format.",
                        attribute_name,
                        reference_name));
      }

      const nvidia::gxf::ColorPlane& color_plane = video_frame_info.color_planes[0];

      const uint64_t element_size = nvidia::gxf::PrimitiveTypeSize(maybe_primitive_type.value());

      shape =
          nvidia::gxf::Shape({static_cast<int32_t>(video_frame_info.height),
                              static_cast<int32_t>(video_frame_info.width),
                              static_cast<int32_t>(color_plane.bytes_per_pixel / element_size)});

      strides.resize(shape.rank());
      strides[2] = static_cast<uint64_t>(color_plane.bytes_per_pixel);
      strides[1] = static_cast<uint64_t>(color_plane.stride);
      strides[0] = strides[1] * static_cast<uint64_t>(video_frame_info.height);
      storage_type = maybe_reference_video_buffer.value()->storage_type();
    } else {
      throw std::runtime_error(
          fmt::format("Attribute '{}': input '{}' has no tensor or video buffer.",
                      attribute_name,
                      reference_name));
    }
  }

  return {shape, strides, storage_type};
}

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
  auto port_info_it = workspace.port_info_.find(port_name_);
  if (port_info_it == workspace.port_info_.end()) {
    auto maybe_entity = workspace.op_input_.receive<gxf::Entity>(port_name_.c_str());
    if (!maybe_entity.has_value()) {
      throw std::runtime_error(
          fmt::format("Failed to receive entity from input port {}.", port_name_));
    }
    // Receive the CUDA stream from the input port (this always should return the same internal
    // stream so we can just assign it to the workspace)
    workspace.cuda_stream_ = workspace.op_input_.receive_cuda_stream(port_name_.c_str());
    port_info_it = workspace.port_info_
                       .insert({port_name_,
                                std::make_unique<CommandWorkspace::PortInfo>(
                                    CommandWorkspace::PortInfo{maybe_entity.value()})})
                       .first;
  }

  // Get the pointer and size of the data
  void* pointer;
  size_t size;

  auto maybe_tensor =
      static_cast<nvidia::gxf::Entity&>(port_info_it->second->entity_)
          .get<nvidia::gxf::Tensor>(item_name_.empty() ? nullptr : item_name_.c_str());
  if (maybe_tensor) {
    // The entity is a tensor
    auto tensor = maybe_tensor.value();
    pointer = tensor->pointer();
    size = tensor->size();
  } else {
    auto maybe_video_buffer =
        static_cast<nvidia::gxf::Entity&>(port_info_it->second->entity_)
            .get<nvidia::gxf::VideoBuffer>(item_name_.empty() ? nullptr : item_name_.c_str());
    if (!maybe_video_buffer) {
      throw std::runtime_error(fmt::format("Attribute 'input': input '{}{}' not found.",
                                           port_name_,
                                           item_name_.empty() ? "" : ":" + item_name_));
    }
    // The entity is a video buffer
    auto video_buffer = maybe_video_buffer.value();
    pointer = video_buffer->pointer();
    size = video_buffer->size();
  }

  // Add the pointer to the CUDA resource pointers
  workspace.cuda_resource_pointers_[resource_name_] = {port_info_it->second.get(), pointer, size};

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

  auto cuda_resource_pointers_it = workspace.cuda_resource_pointers_.find(resource_name_);
  if (cuda_resource_pointers_it == workspace.cuda_resource_pointers_.end()) {
    throw std::runtime_error(
        fmt::format("Attribute 'output': output '{}' not found, "
                    "did you forget to allocate memory for it?",
                    port_name_));
  }

  auto& port_info = cuda_resource_pointers_it->second.port_info_;

  // Make sure to emit the entity only once (we might have several items in the entity)
  if (!port_info->has_been_emitted_) {
    workspace.op_output_.emit(port_info->entity_, port_name_.c_str());
    port_info->has_been_emitted_ = true;
  }
}

CommandAlloc::CommandAlloc(const std::string& port_name, const std::string& item_name,
                           const std::string& resource_name, const std::string& reference_name,
                           uint32_t size_x, uint32_t size_y, uint32_t size_z,
                           const std::string& element_type, uint32_t element_count,
                           const Parameter<std::shared_ptr<Allocator>>& allocator,
                           size_t parameter_offset)
    : port_name_(port_name),
      item_name_(item_name),
      resource_name_(resource_name),
      reference_name_(reference_name),
      size_x_(size_x),
      size_y_(size_y),
      size_z_(size_z),
      element_type_(element_type),
      element_count_(element_count),
      allocator_(allocator),
      parameter_offset_(parameter_offset) {}

void CommandAlloc::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandAlloc: {}{} {} {} {}x{}x{} {}{}",
                     port_name_,
                     item_name_.empty() ? "" : ":" + item_name_,
                     resource_name_,
                     reference_name_,
                     size_x_,
                     size_y_,
                     size_z_,
                     element_type_,
                     element_count_);

  nvidia::gxf::Shape alloc_shape;
  std::string swizzle;
  nvidia::gxf::MemoryStorageType storage_type;

  if (!reference_name_.empty()) {
    // If we have a reference name, we need to get the shape of the reference resource
    std::string reference_remainder;
    std::tie(reference_remainder, swizzle) = split(reference_name_, '.');
    auto [reference_shape, reference_strides, reference_storage_type] =
        get_reference_specs(workspace, reference_remainder, "alloc/alloc_size_of");
    alloc_shape = reference_shape;
    storage_type = reference_storage_type;
  } else {
    std::vector<int32_t> dimensions;
    if (size_z_ > 1) {
      dimensions.push_back(static_cast<int32_t>(size_z_));
    }
    if (size_y_ > 1) {
      dimensions.push_back(static_cast<int32_t>(size_y_));
    }
    if (size_x_ > 1) {
      dimensions.push_back(static_cast<int32_t>(size_x_));
    }
    dimensions.push_back(element_count_);

    // Create a new shape and set the storage type to device
    alloc_shape = nvidia::gxf::Shape(dimensions);
    storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  }

  // Create the port info if it does not exist
  auto port_info_it = workspace.port_info_.find(port_name_);
  if (port_info_it == workspace.port_info_.end()) {
    holoscan::expected<gxf::Entity, holoscan::RuntimeError> maybe_entity =
        gxf::Entity::New(&workspace.context_);
    if (!maybe_entity.has_value()) {
      throw std::runtime_error(
          fmt::format("Failed to create entity for output port {}.", port_name_));
    }
    port_info_it = workspace.port_info_
                       .insert({port_name_,
                                std::make_unique<CommandWorkspace::PortInfo>(
                                    CommandWorkspace::PortInfo{maybe_entity.value()})})
                       .first;
  }

  // Add a tensor to the entity
  auto tensor = static_cast<nvidia::gxf::Entity&>(port_info_it->second->entity_)
                    .add<nvidia::gxf::Tensor>(item_name_.empty() ? nullptr : item_name_.c_str())
                    .value();

  // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<Allocator>
  auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      workspace.context_.context(), allocator_.get()->gxf_cid());

  std::vector<int32_t> result;
  if (!swizzle.empty()) {
    // Apply the swizzle to the reference resource shape
    const nvidia::gxf::Shape swizzled_shape = swizzle_shape(alloc_shape, swizzle);
    for (int i = 0; i < swizzled_shape.rank(); ++i) {
      result.push_back(swizzled_shape.dimension(i));
    }
  } else {
    // Copy anything but the last dimension which is the element count
    if (alloc_shape.rank() > 0) {
      for (int i = 0; i < alloc_shape.rank() - 1; ++i) {
        result.push_back(alloc_shape.dimension(i));
      }
    }
  }
  // Replace the element count of the reference resource with the element count of the shader
  // resource
  result.push_back(element_count_);

  alloc_shape = nvidia::gxf::Shape(result);

  // Reshape the tensor to the size of the reference resource
  if (element_type_ == "int8") {
    tensor->reshape<int8_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "uint8") {
    tensor->reshape<uint8_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "int16") {
    tensor->reshape<int16_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "uint16") {
    tensor->reshape<uint16_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "int32") {
    tensor->reshape<int32_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "uint32") {
    tensor->reshape<uint32_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "int64") {
    tensor->reshape<int64_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "uint64") {
    tensor->reshape<uint64_t>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "float32") {
    tensor->reshape<float>(alloc_shape, storage_type, gxf_allocator.value());
  } else if (element_type_ == "float64") {
    tensor->reshape<double>(alloc_shape, storage_type, gxf_allocator.value());
  } else {
    throw std::runtime_error(fmt::format("Element type '{}' not supported.", element_type_));
  }

  void* const pointer = tensor->pointer();

  // Add the pointer to the CUDA resource pointers
  workspace.cuda_resource_pointers_[resource_name_] = {
      port_info_it->second.get(), pointer, tensor->size()};

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

  // Get the shape of the reference resource and copy it to the shader parameters array
  auto [reference_remainder, swizzle] = split(reference_port_name_, '.');
  auto [shape, strides, storage_type] =
      get_reference_specs(workspace, reference_remainder, "size_of");
  const auto swizzled_shape = swizzle_shape(shape, swizzle);

  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(uint3));
  uint3* param_data =
      reinterpret_cast<uint3*>(workspace.shader_parameters_.data() + parameter_offset_);
  // If we don't swizzle, we need to skip the element count, else the user is responsible for
  // setting the swizzle correctly
  *param_data = shape_to_uint3(swizzled_shape, swizzle.empty());
}

CommandStrideOf::CommandStrideOf(const std::string& parameter_name,
                                 const std::string& reference_port_name, size_t parameter_offset)
    : parameter_name_(parameter_name),
      reference_port_name_(reference_port_name),
      parameter_offset_(parameter_offset) {}

void CommandStrideOf::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandStrideOf: {} {}", parameter_name_, reference_port_name_);

  // Get the strides of the reference resource and copy it to the shader parameters array
  auto [reference_shape, reference_strides, reference_storage_type] =
      get_reference_specs(workspace, reference_port_name_, "strides_of");

  workspace.shader_parameters_.resize(parameter_offset_ + sizeof(ulonglong3));
  ulonglong3 strides = make_ulonglong3(1, 1, 1);
  if (reference_strides.size() > 1) {
    strides.x = reference_strides[reference_strides.size() - 2];
    if (reference_strides.size() > 2) {
      strides.y = reference_strides[reference_strides.size() - 3];
      if (reference_strides.size() > 3) {
        strides.z = reference_strides[reference_strides.size() - 4];
        if (reference_strides.size() > 4) {
          throw std::runtime_error(fmt::format(
              "Attribute 'strides_of': Only tensors of rank <= 4 are supported, found rank '{}'.",
              reference_shape.rank()));
        }
      } else {
        strides.z = strides.y;
      }
    } else {
      strides.y = strides.x;
    }
  }
  memcpy(workspace.shader_parameters_.data() + parameter_offset_, &strides, sizeof(ulonglong3));
}

void CommandReceiveCudaStream::execute(CommandWorkspace& workspace) {
  // If there are inputs, we receive the CUDA stream from the input port, else we allocate a new
  // stream
  if (!workspace.cuda_stream_) {
    auto maybe_cuda_stream = workspace.context_.allocate_cuda_stream();
    if (!maybe_cuda_stream) {
      throw std::runtime_error(
          fmt::format("Failed to allocate CUDA stream: {}", maybe_cuda_stream.error().what()));
    }
    workspace.cuda_stream_ = maybe_cuda_stream.value();
  }
}

CommandLaunch::CommandLaunch(const std::string& name, SlangShaderCompiler* shader_compiler,
                             dim3 thread_group_size, const std::string& invocations_size_of_name,
                             dim3 invocations)
    : name_(name),
      shader_compiler_(shader_compiler),
      thread_group_size_(thread_group_size),
      invocations_size_of_name_(invocations_size_of_name),
      invocations_(invocations) {
  // Get the kernel for the entry point
  kernel_ = shader_compiler_->get_kernel(name_);

  if ((thread_group_size_.x == 1) && (thread_group_size_.y == 1) && (thread_group_size_.z == 1)) {
    int min_threads_per_block, min_blocks_per_grid;
    CUDA_CALL(
        cudaOccupancyMaxPotentialBlockSize(&min_blocks_per_grid, &min_threads_per_block, kernel_));
    /// @todo this assumes a 2D kernel, add a way for the user to specify dimensionality
    thread_group_size_ = dim3(1, 1, 1);
    while (static_cast<int>(thread_group_size_.x * thread_group_size_.y * 2) <=
           min_threads_per_block) {
      if (thread_group_size_.x > thread_group_size_.y) {
        thread_group_size_.y *= 2;
      } else {
        thread_group_size_.x *= 2;
      }
    }
    HOLOSCAN_LOG_DEBUG("CommandLaunch: Autodetected thread group size: {}x{}x{}",
                       thread_group_size_.x,
                       thread_group_size_.y,
                       thread_group_size_.z);
  }
}

void CommandLaunch::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandLaunch: {}", name_);

  // Setup the invocation size
  dim3 invocations;
  if (!invocations_size_of_name_.empty()) {
    auto [invocations_size_of_remainder, swizzle] = split(invocations_size_of_name_, '.');

    auto [reference_shape, reference_strides, reference_storage_type] =
        get_reference_specs(workspace, invocations_size_of_remainder, "invocations_size_of");
    const auto swizzled_shape = swizzle_shape(reference_shape, swizzle);
    // If we don't swizzle, we need to skip the element count, else the user is responsible for
    // setting the swizzle correctly
    invocations = shape_to_uint3(swizzled_shape, swizzle.empty());
  } else {
    invocations = invocations_;
  }

  // limit thread group size to invocations
  const dim3 actual_thread_group_size = dim3(std::min(thread_group_size_.x, invocations.x),
                                             std::min(thread_group_size_.y, invocations.y),
                                             std::min(thread_group_size_.z, invocations.z));

  // Calculate the grid size
  const dim3 grid_size =
      dim3((invocations.x + actual_thread_group_size.x - 1) / actual_thread_group_size.x,
           (invocations.y + actual_thread_group_size.y - 1) / actual_thread_group_size.y,
           (invocations.z + actual_thread_group_size.z - 1) / actual_thread_group_size.z);

  // Set the global params
  if (!workspace.shader_parameters_.empty()) {
    shader_compiler_->update_global_params(
        name_, workspace.shader_parameters_, workspace.cuda_stream_);
  }

  // Launch the kernel
  CUDA_CALL(cudaLaunchKernel(
      kernel_, grid_size, actual_thread_group_size, nullptr, 0, workspace.cuda_stream_));
}

CommandZeros::CommandZeros(const std::string& resource_name) : resource_name_(resource_name) {}

void CommandZeros::execute(CommandWorkspace& workspace) {
  HOLOSCAN_LOG_DEBUG("CommandZeros: {}", resource_name_);

  // Get the pointer and size of the data
  auto cuda_resource_pointers_it = workspace.cuda_resource_pointers_.find(resource_name_);
  if (cuda_resource_pointers_it == workspace.cuda_resource_pointers_.end()) {
    throw std::runtime_error(
        fmt::format("Attribute 'zeros': resource '{}' not found, "
                    "did you forget to allocate memory for it?",
                    resource_name_));
  }

  // Initialize the data to zero
  CUDA_CALL(cudaMemsetAsync(cuda_resource_pointers_it->second.pointer_,
                            0,
                            cuda_resource_pointers_it->second.size_,
                            workspace.cuda_stream_));
}

}  // namespace holoscan::ops
