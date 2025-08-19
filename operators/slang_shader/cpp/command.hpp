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

#ifndef COMMAND_HPP
#define COMMAND_HPP

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>

#include "slang_utils.hpp"

namespace holoscan::ops {

class SlangShaderCompiler;

/**
 * @brief Workspace for command execution containing input/output contexts and shared resources
 *
 * This class provides a centralized workspace that holds all the necessary context
 * and resources needed for command execution, including input/output contexts,
 * CUDA streams, entity mappings, and shader parameters.
 */
class CommandWorkspace {
 public:
  /**
   * @brief Constructs a command workspace with the provided contexts
   *
   * @param op_input Reference to the operator input context
   * @param op_output Reference to the operator output context
   * @param context Reference to the execution context
   */
  CommandWorkspace(InputContext& op_input, OutputContext& op_output, ExecutionContext& context);

  InputContext& op_input_;     ///< Operator input context
  OutputContext& op_output_;   ///< Operator output context
  ExecutionContext& context_;  ///< Execution context

  /// Port information
  struct PortInfo {
    gxf::Entity entity_;
    bool has_been_emitted_ = false;
  };

  /**
   * @brief Mapping of port names to port information
   *
   * This map is used to store information about input and output ports. This will be filled when
   * input and alloc commands are executed.
   */
  std::map<std::string, std::unique_ptr<PortInfo>> port_info_;

  std::vector<uint8_t> shader_parameters_;  ///< Buffer for shader parameter data

  /// Resource information
  struct ResourceInfo {
    PortInfo* port_info_ = nullptr;
    void* pointer_ = nullptr;
    size_t size_ = 0;
  };

  /**
   * @brief Mapping of resource names to CUDA resource information
   *
   * This map is used to store information about CUDA resources, including the port information,
   * pointer, and size. This will be filled when input and alloc commands are executed.
   * It is used on output commands to get the pointer to the port info and on the zeros command to
   * get the pointer to the CUDA resource.
   */
  std::map<std::string, ResourceInfo>
      cuda_resource_pointers_;  ///< Mapping of resource names to CUDA resource information

  cudaStream_t cuda_stream_ = 0;  ///< CUDA stream for asynchronous operations
};

/**
 * @brief Abstract base class for all command types
 *
 * Commands implement the command pattern to encapsulate operations that can be
 * executed on a CommandWorkspace. Each command represents a specific operation
 * such as input/output handling, parameter management, or kernel launching.
 */
class Command {
 public:
  virtual ~Command() = default;

  /**
   * @brief Executes the command using the provided workspace
   *
   * @param workspace The workspace containing all necessary context and resources
   */
  virtual void execute(CommandWorkspace& workspace) = 0;
};

/**
 * @brief Command for handling input port operations
 *
 * This command is responsible for processing input data from specified ports
 * and storing the results in the workspace for use by subsequent commands.
 */
class CommandInput : public Command {
 public:
  /**
   * @brief Constructs an input command
   *
   * @param port_name Name of the input port to read from.
   * @param item_name Name of the item to get from the port. If empty, the first item is read.
   * @param resource_name Name of the resource to store the input data
   * @param parameter_offset Offset in the parameter buffer for this input
   */
  CommandInput(const std::string& port_name, const std::string& item_name,
               const std::string& resource_name, size_t parameter_offset);
  CommandInput() = delete;

  /**
   * @brief Executes the input command
   *
   * Receives data from the specified input port and stores it in the workspace
   * at the specified resource location and parameter offset.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string port_name_;      ///< Name of the input port
  const std::string item_name_;      ///< Name of the item to read from the port
  const std::string resource_name_;  ///< Name of the resource to store data
  const size_t parameter_offset_;    ///< Offset in parameter buffer
};

/**
 * @brief Command for handling output port operations
 *
 * This command is responsible for emitting processed data to specified output ports
 * from the workspace resources.
 */
class CommandOutput : public Command {
 public:
  /**
   * @brief Constructs an output command
   *
   * @param port_name Name of the output port to write to
   * @param item_name Name of the item to write to the port. If empty, the first item is written.
   * @param resource_name Name of the resource containing the output data
   */
  explicit CommandOutput(const std::string& port_name, const std::string& item_name,
                         const std::string& resource_name);
  CommandOutput() = delete;

  /**
   * @brief Executes the output command
   *
   * Emits data from the specified resource to the output port.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

  const std::string& port_name() const { return port_name_; }
  const std::string& item_name() const { return item_name_; }

 private:
  const std::string port_name_;      ///< Name of the output port
  const std::string item_name_;      ///< Name of the item to write to the port
  const std::string resource_name_;  ///< Name of the resource containing data
};

/**
 * @brief Command for allocating memory with size information
 *
 * This command handles memory allocation operations where the size is determined
 * dynamically from another resource or specified explicitly.
 */
class CommandAlloc : public Command {
 public:
  /**
   * @brief Constructs an allocation command with size information
   *
   * @param port_name Name of the port to allocate memory for
   * @param item_name Name of the item to allocate memory for
   * @param resource_name Name of the resource to allocate memory for
   * @param reference_name Name of the port containing size information
   * @param size_x Number of elements in the X dimension
   * @param size_y Number of elements in the Y dimension
   * @param size_z Number of elements in the Z dimension
   * @param element_type Element type of the data to allocate memory for
   * @param element_count Number of elements in the data to allocate memory for
   * @param allocator Reference to the allocator parameter
   * @param parameter_offset Offset in the parameter buffer
   */
  CommandAlloc(const std::string& port_name, const std::string& item_name,
               const std::string& resource_name, const std::string& reference_name, uint32_t size_x,
               uint32_t size_y, uint32_t size_z, const std::string& element_type,
               uint32_t element_count, const Parameter<std::shared_ptr<Allocator>>& allocator,
               size_t parameter_offset);
  CommandAlloc() = delete;

  /**
   * @brief Executes the allocation command
   *
   * Allocates memory using the specified allocator with size determined
   * from the reference_port_name resource.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string port_name_;       ///< Name of the port to allocate memory for
  const std::string item_name_;       ///< Name of the item to allocate memory for
  const std::string resource_name_;   ///< Name of the resource to allocate memory for
  const std::string reference_name_;  ///< Name of the port containing size information
  const uint32_t size_x_;             ///< Number of elements in the X dimension
  const uint32_t size_y_;             ///< Number of elements in the Y dimension
  const uint32_t size_z_;             ///< Number of elements in the Z dimension
  const std::string element_type_;    ///< Element type of the data to allocate memory for
  const uint32_t element_count_;      ///< Number of elements in the data to allocate memory for
  const Parameter<std::shared_ptr<Allocator>>& allocator_;  ///< Allocator reference
  const size_t parameter_offset_;                           ///< Offset in parameter buffer
};

/**
 * @brief Command for managing parameter values
 *
 * This command handles parameter storage and retrieval operations, providing
 * type-safe access to parameter values in the workspace.
 */
class CommandParameter : public Command {
 public:
  /**
   * @brief Constructs a parameter command
   *
   * @tparam typeT The type of the parameter
   * @param spec The operator specification to register the parameter with
   * @param param Pointer to the parameter object
   * @param name Name of the parameter
   * @param parameter_offset Offset in the parameter buffer
   */
  template <typename typeT>
  explicit CommandParameter(OperatorSpec& spec, Parameter<typeT>* param, const std::string& name,
                            size_t parameter_offset)
      : value_(reinterpret_cast<void*>(param),
               [](void* value) { delete static_cast<Parameter<typeT>*>(value); }),
        parameter_offset_(parameter_offset),
        size_(sizeof(typeT)) {
    // Split the name into parameter name and default value
    auto [parameter_name, default_value_str] = split(name, '=');
    name_ = parameter_name;

    // If the default value is not provided, use the default constructor, else convert the string to
    // the type
    typeT default_value;
    if (default_value_str.empty()) {
      default_value = typeT();
    } else {
      default_value = from_string<typeT>(default_value_str);
    }

    // Create the parameter in the operator spec
    spec.param<typeT>(*param, name_.c_str(), "N/A", "N/A", default_value);

    get_value_ = [](void* value) -> void* {
      return reinterpret_cast<void*>(&static_cast<Parameter<typeT>*>(value)->get());
    };
  }
  CommandParameter() = delete;

  /**
   * @brief Executes the parameter command
   *
   * Stores or retrieves parameter values in the workspace parameter buffer.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  std::string name_;                   ///< Parameter name
  const std::shared_ptr<void> value_;  ///< Parameter value storage
  const size_t parameter_offset_;      ///< Offset in parameter buffer
  const size_t size_;                  ///< Size of the parameter in bytes

  std::function<void*(void* value)> get_value_;  ///< Function to get parameter value
};

/**
 * @brief Command for handling size-of operations
 *
 * Retrieves the size of the specified resource and stores it at the
 * name location in the parameter buffer.
 */
class CommandSizeOf : public Command {
 public:
  /**
   * @brief Constructs a size-of command
   *
   * @param parameter_name Name of the parameter where the size information will be stored
   * @param reference_port_name Name of the port containing size information
   * @param parameter_offset Offset in the parameter buffer
   */
  CommandSizeOf(const std::string& parameter_name, const std::string& reference_port_name,
                size_t parameter_offset);
  CommandSizeOf() = delete;

  /**
   * @brief Executes the size-of command
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string
      parameter_name_;  ///< Name of the parameter where the size information will be stored
  const std::string reference_port_name_;  ///< Name of the port containing size information
  const size_t parameter_offset_;          ///< Offset in parameter buffer
};

/**
 * @brief Command for handling stride-of operations
 *
 * Retrieves the strides of the specified resource and stores it at the
 * name location in the parameter buffer.
 */
class CommandStrideOf : public Command {
 public:
  /**
   * @brief Constructs a size-of command
   *
   * @param parameter_name Name of the parameter where the size information will be stored
   * @param reference_port_name Name of the port containing size information
   * @param parameter_offset Offset in the parameter buffer
   */
  CommandStrideOf(const std::string& parameter_name, const std::string& reference_port_name,
                  size_t parameter_offset);
  CommandStrideOf() = delete;

  /**
   * @brief Executes the size-of command
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string
      parameter_name_;  ///< Name of the parameter where the size information will be stored
  const std::string reference_port_name_;  ///< Name of the port containing size information
  const size_t parameter_offset_;          ///< Offset in parameter buffer
};

/**
 * @brief Command for receiving a CUDA stream
 *
 * This command receives a CUDA stream from the operator input context
 * and stores it in the workspace for use by subsequent commands.
 */
class CommandReceiveCudaStream : public Command {
 public:
  /**
   * @brief Constructs a CUDA stream receive command
   */
  CommandReceiveCudaStream() = default;

  /**
   * @brief Executes the CUDA stream receive command
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
};

/**
 * @brief Command for launching CUDA kernels
 *
 * This command handles the execution of CUDA kernels with specified
 * invocation and thread group dimensions. The `thread_group_size` parameter defines the number of
 * threads per thread group, if it is (1,1,1) the best thread group size is automatically selected.
 * The `invocations` parameter defines the total number of invocations.
 */
class CommandLaunch : public Command {
 public:
  /**
   * @brief Constructs a kernel launch command
   *
   * @param name Name of the kernel
   * @param shader_compiler Pointer to the SlangShaderCompiler object
   * @param thread_group_size CUDA thread group dimensions (threads per thread group).
   * @param invocations_size_of_name Name of parameter containing invocation size
   * @param invocations total number of invocations
   */
  CommandLaunch(const std::string& name, SlangShaderCompiler* shader_compiler,
                dim3 thread_group_size, const std::string& invocations_size_of_name,
                dim3 invocations);
  CommandLaunch() = delete;

  /**
   * @brief Executes the kernel launch command
   *
   * Compiles and launches the CUDA kernel with the specified parameters
   * and invocation/thread group dimensions.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string name_;                      ///< Kernel name
  SlangShaderCompiler* const shader_compiler_;  ///< Pointer to the shader compiler object
  dim3 thread_group_size_;  ///< CUDA thread group dimensions (threads per thread group).
  const std::string invocations_size_of_name_;  ///< Name of invocation size parameter
  const dim3 invocations_;                      ///< Total number of invocations
  cudaKernel_t kernel_ = nullptr;               ///< Compiled CUDA kernel handle
};

/**
 * @brief Command for initializing resources to zero
 *
 * This command initializes the specified resource to zero.
 */
class CommandZeros : public Command {
 public:
  /**
   * @brief Constructs a zeros command
   *
   * @param resource_name Name of the resource to initialize to zero
   */
  explicit CommandZeros(const std::string& resource_name);
  CommandZeros() = delete;

  /**
   * @brief Executes the zeros command
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string resource_name_;  ///< Name of the resource to initialize to zero
};

}  // namespace holoscan::ops

#endif  // COMMAND_HPP
