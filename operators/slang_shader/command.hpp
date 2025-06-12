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

namespace holoscan::ops {

class SlangShader;

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

  std::map<std::string, gxf::Entity> entities_;  ///< Mapping of entity names to GXF entities

  std::vector<uint8_t> shader_parameters_;  ///< Buffer for shader parameter data

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
   * @param port_name Name of the input port to read from
   * @param resource_name Name of the resource to store the input data
   * @param parameter_offset Offset in the parameter buffer for this input
   */
  CommandInput(const std::string& port_name, const std::string& resource_name,
               size_t parameter_offset);
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
   * @param resource_name Name of the resource containing the output data
   */
  explicit CommandOutput(const std::string& port_name, const std::string& resource_name);
  CommandOutput() = delete;

  /**
   * @brief Executes the output command
   *
   * Emits data from the specified resource to the output port.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string port_name_;      ///< Name of the output port
  const std::string resource_name_;  ///< Name of the resource containing data
};

/**
 * @brief Command for allocating memory with size information
 *
 * This command handles memory allocation operations where the size is determined
 * dynamically from another resource.
 */
class CommandAllocSizeOf : public Command {
 public:
  /**
   * @brief Constructs an allocation command with size information
   *
   * @param name Name of the allocation
   * @param size_of_name Name of the resource containing size information
   * @param allocator Reference to the allocator parameter
   * @param parameter_offset Offset in the parameter buffer
   */
  CommandAllocSizeOf(const std::string& name, const std::string& size_of_name,
                     const Parameter<std::shared_ptr<Allocator>>& allocator,
                     size_t parameter_offset);
  CommandAllocSizeOf() = delete;

  /**
   * @brief Executes the allocation command
   *
   * Allocates memory using the specified allocator with size determined
   * from the size_of_name resource.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string name_;                                  ///< Name of the allocation
  const std::string size_of_name_;                          ///< Name of size resource
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
      : name_(name),
        value_(reinterpret_cast<void*>(param),
               [](void* value) { delete static_cast<Parameter<typeT>*>(value); }),
        parameter_offset_(parameter_offset),
        size_(sizeof(typeT)) {
    // Create the parameter in the operator spec
    spec.param<typeT>(*param, name_.c_str());

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
  const std::string name_;             ///< Parameter name
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
   * @param name Name where the size information will be stored
   * @param size_of_name Name of the resource to get size for
   * @param parameter_offset Offset in the parameter buffer
   */
  CommandSizeOf(const std::string& name, const std::string& size_of_name, size_t parameter_offset);
  CommandSizeOf() = delete;

  /**
   * @brief Executes the size-of command
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string name_;          ///< Name where the size information will be stored
  const std::string size_of_name_;  ///< Name of the resource to get size for
  const size_t parameter_offset_;   ///< Offset in parameter buffer
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
 * grid and block dimensionsß.
 */
class CommandLaunch : public Command {
 public:
  /**
   * @brief Constructs a kernel launch command
   *
   * @param name Name of the kernel
   * @param shader Pointer to the SlangShader object
   * @param block_size CUDA block dimensions
   * @param grid_size_of_name Name of parameter containing grid size
   * @param grid_size CUDA grid dimensions
   */
  CommandLaunch(const std::string& name, SlangShader* shader, dim3 block_size,
                const std::string& grid_size_of_name, dim3 grid_size);
  CommandLaunch() = delete;

  /**
   * @brief Executes the kernel launch command
   *
   * Compiles and launches the CUDA kernel with the specified parameters
   * and grid/block dimensions.
   *
   * @param workspace The workspace to operate on
   */
  void execute(CommandWorkspace& workspace) override;

 private:
  const std::string name_;               ///< Kernel name
  SlangShader* const shader_;            ///< Pointer to the shader object
  const dim3 block_size_;                ///< CUDA block dimensions
  const std::string grid_size_of_name_;  ///< Name of grid size parameter
  const dim3 grid_size_;                 ///< CUDA grid dimensions
  cudaKernel_t kernel_ = nullptr;        ///< Compiled CUDA kernel handle
};

}  // namespace holoscan::ops

#endif  // COMMAND_HPP
