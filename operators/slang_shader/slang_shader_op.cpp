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

#include "slang_shader_op.hpp"

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "command.hpp"
#include "slang_shader.hpp"
#include "slang_utils.hpp"

#include "holoscan_slang.hpp"

namespace holoscan::ops {

// Implementation struct containing all Slang-related details
class SlangShaderOp::Impl {
 public:
  Impl() {
    // First we need to create slang global session to work with the Slang API.
    SLANG_CALL(slang::createGlobalSession(global_session_.writeRef()));

    // Create Session
    slang::SessionDesc sessionDesc;

    // Set up target description for PTX
    slang::TargetDesc targetDesc;
    targetDesc.format = SLANG_PTX;

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    // Create session
    SLANG_CALL(global_session_->createSession(sessionDesc, session_.writeRef()));

    // Load module from source string
    Slang::ComPtr<ISlangBlob> diagnostics_blob;

    holoscan_module_ = session_->loadModuleFromSourceString(
        "holoscan", "holoscan.slang", holoscan_slang, diagnostics_blob.writeRef());
    // Check for compilation errors
    SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);
    if (!holoscan_module_) {
      throw std::runtime_error("Failed to compile Slang module");
    }
  }

  Slang::ComPtr<slang::ISession> session_;

  std::unique_ptr<SlangShader> shader_;

  std::list<std::unique_ptr<Command>> pre_launch_commands_;
  std::list<std::unique_ptr<Command>> launch_commands_;
  std::list<std::unique_ptr<Command>> post_launch_commands_;

 private:
  // Slang compilation components
  Slang::ComPtr<slang::IGlobalSession> global_session_;
  Slang::ComPtr<slang::IModule> holoscan_module_;
};

namespace {

/**
 * Creates a CommandParameter object based on the parameter type from Slang reflection data.
 *
 * This function analyzes the scalar type from the Slang parameter reflection and creates
 * the appropriate CommandParameter with the corresponding C++ type. It supports various
 * scalar types including bool, int8/16/32/64, uint8/16/32/64, float32, and float64.
 *
 * @param spec The OperatorSpec to add the parameter to
 * @param param_name The name of the parameter to create
 * @param parameter The JSON reflection data containing parameter type and binding information
 * @return A unique_ptr to the created CommandParameter, or nullptr if the type is unsupported
 */
std::unique_ptr<CommandParameter> create_command_parameter(OperatorSpec& spec,
                                                           const std::string& param_name,
                                                           const nlohmann::json& parameter) {
  if (parameter["type"]["scalarType"] == "bool") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<bool>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int32") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<int32_t>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint32") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<uint32_t>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int64") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<int64_t>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint64") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<uint64_t>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "float32") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<float>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "float64") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<double>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int8") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<int8_t>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint8") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<uint8_t>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int16") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<int16_t>(), param_name, parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint16") {
    return std::make_unique<CommandParameter>(
        spec, new Parameter<uint16_t>(), param_name, parameter["binding"]["offset"]);
  }

  return nullptr;
}

}  // namespace

void SlangShaderOp::initialize() {
  // Add the allocator to the operator so that it is initialized, this is needed because the
  // parameter has a default allocator
  add_arg(allocator_.default_value());

  // Call the base class initialize function
  Operator::initialize();
}

void SlangShaderOp::setup(OperatorSpec& spec) {
  assert(!impl_);
  impl_ = std::make_shared<Impl>();

  spec.param(shader_source_, "shader_source", "Shader source string.", "Shader source string.");

  spec.param(
      shader_source_file_, "shader_source_file", "Shader source file.", "Shader source file.");

  spec.param(allocator_,
             "allocator",
             "Allocator for output buffers.",
             "Allocator for output buffers.",
             std::static_pointer_cast<Allocator>(
                 fragment()->make_resource<UnboundedAllocator>("allocator")));

  // Add a CUDA stream pool
  add_arg(fragment()->make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 0));

  // We need the shader source to build the input and output ports and the parameters, so check
  // the argument list and get them
  std::string shader_source, shader_source_file;
  for (auto&& arg : args()) {
    if (arg.name() == "shader_source") {
      shader_source = std::any_cast<std::string>(arg.value());
    } else if (arg.name() == "shader_source_file") {
      shader_source_file = std::any_cast<std::string>(arg.value());
    }
  }

  if (shader_source.empty() && shader_source_file.empty()) {
    throw std::runtime_error("Either 'shader_source' or 'shader_source_file' must be set");
  }

  if (!shader_source.empty() && !shader_source_file.empty()) {
    throw std::runtime_error("Both 'shader_source' and 'shader_source_file' cannot be set");
  }

  // load from file
  if (!shader_source_file.empty()) {
    std::ifstream in_stream(shader_source_file);
    if (!in_stream.is_open()) {
      throw std::runtime_error(fmt::format("Failed to open shader file '{}'", shader_source_file));
    }
    std::stringstream shader_string;
    shader_string << in_stream.rdbuf();
    shader_source = shader_string.str();
  }

  impl_->shader_ = std::make_unique<SlangShader>(impl_->session_, shader_source);

  const nlohmann::json reflection = impl_->shader_->get_reflection();

  // Get Slang parameters and setup inputs, outputs and parameters
  for (auto& parameter : reflection["parameters"]) {
    if (!parameter.contains("userAttribs")) {
      continue;
    }

    // Check the user attributes and check for Holoscan attributes
    for (auto& user_attrib : parameter["userAttribs"]) {
      // Ignore non-Holoscan attributes
      const std::string user_attrib_name = user_attrib["name"];
      const std::string holoscan_prefix = "holoscan_";
      if (user_attrib_name.find(holoscan_prefix) != 0) {
        continue;
      }

      const std::string attrib_name = user_attrib_name.substr(holoscan_prefix.size());

      if (attrib_name == "input") {
        // input
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }
        const std::string input_name = user_attrib["arguments"].at(0);
        spec.input<gxf::Entity>(input_name);
        impl_->pre_launch_commands_.push_back(std::make_unique<CommandInput>(
            input_name, parameter["name"], parameter["binding"]["offset"]));
      } else if (attrib_name == "output") {
        // output
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }
        const std::string output_name = user_attrib["arguments"].at(0);
        spec.output<gxf::Entity>(output_name);
        impl_->post_launch_commands_.push_back(
            std::make_unique<CommandOutput>(output_name, parameter["name"]));
      } else if (attrib_name == "alloc_size_of") {
        // size_of
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }
        const std::string size_of_name = user_attrib["arguments"].at(0);
        impl_->pre_launch_commands_.push_back(std::make_unique<CommandAllocSizeOf>(
            parameter["name"], size_of_name, allocator_, parameter["binding"]["offset"]));
      } else if (attrib_name == "parameter") {
        // parameter
        if ((parameter["binding"]["kind"] != "uniform") ||
            (parameter["type"]["kind"] != "scalar")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports scalar uniforms only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }

        const std::string param_name = user_attrib["arguments"].at(0);
        auto command_parameter = create_command_parameter(spec, param_name, parameter);
        if (!command_parameter) {
          throw std::runtime_error(
              fmt::format("Attribute '{}' unsupported scalar type '{}' for parameter '{}'.",
                          attrib_name,
                          parameter["type"]["scalarType"].get<std::string>(),
                          parameter["name"].get<std::string>()));
        }
        impl_->pre_launch_commands_.push_back(std::move(command_parameter));
      } else if (attrib_name == "size_of") {
        // size_of
        if ((parameter["binding"]["kind"] != "uniform") ||
            (parameter["type"]["kind"] != "vector") || (parameter["type"]["elementCount"] != 3) ||
            (parameter["type"]["elementType"]["kind"] != "scalar") ||
            (parameter["type"]["elementType"]["scalarType"] != "int32")) {
          throw std::runtime_error(
              fmt::format("Attribute '{}' supports a three component int32 vector (`int3`) "
                          "uniforms only and cannot be applied to '{}'.",
                          attrib_name,
                          parameter["name"].get<std::string>()));
        }
        const std::string size_of_name = user_attrib["arguments"].at(0);
        impl_->pre_launch_commands_.push_back(std::make_unique<CommandSizeOf>(
            parameter["name"], size_of_name, parameter["binding"]["offset"]));
      } else {
        throw std::runtime_error("Unknown user attribute: " + user_attrib_name);
      }
    }
  }

  // After all the inputs are handled, receive the CUDA stream
  impl_->pre_launch_commands_.push_back(std::make_unique<CommandReceiveCudaStream>());

  // Get Slang entry points and launch commands
  for (auto& entry_point : reflection["entryPoints"]) {
    if (entry_point["stage"] != "compute") {
      throw std::runtime_error("Only compute entry points are supported");
    }

    std::string grid_size_of_name;
    dim3 grid_size(1, 1, 1);

    // Check the user attributes and check for Holoscan attributes
    for (auto& user_attrib : entry_point["userAttribs"]) {
      const std::string user_attrib_name = user_attrib["name"];
      const std::string holoscan_prefix = "holoscan_";
      if (user_attrib_name.find(holoscan_prefix) != 0) {
        continue;
      }

      const std::string attrib_name = user_attrib_name.substr(holoscan_prefix.size());

      if (attrib_name == "grid_size_of") {
        if (user_attrib["arguments"].empty()) {
          throw std::runtime_error("Attribute 'grid_size_of' requires an argument");
        }
        grid_size_of_name = user_attrib["arguments"].at(0);
      } else if (attrib_name == "grid_size") {
        if (user_attrib["arguments"].empty()) {
          throw std::runtime_error("Attribute 'grid_size' requires an argument");
        }
        grid_size.x = user_attrib["arguments"].at(0);
        if (user_attrib["arguments"].size() > 1) {
          grid_size.y = user_attrib["arguments"].at(1);
          if (user_attrib["arguments"].size() > 2) {
            grid_size.z = user_attrib["arguments"].at(2);
          }
        }
      } else {
        throw std::runtime_error("Unknown user attribute: " + user_attrib_name);
      }
    }

    dim3 block_size(1, 1, 1);
    if (entry_point.contains("threadGroupSize")) {
      block_size.x = entry_point["threadGroupSize"][0];
      block_size.y = entry_point["threadGroupSize"][1];
      block_size.z = entry_point["threadGroupSize"][2];
    }

    // And then launch the kernel
    impl_->launch_commands_.push_back(std::make_unique<CommandLaunch>(
        entry_point["name"], impl_->shader_.get(), block_size, grid_size_of_name, grid_size));
  }
}

void SlangShaderOp::compute(InputContext& op_input, OutputContext& op_output,
                            ExecutionContext& context) {
  CommandWorkspace workspace(op_input, op_output, context);

  // Execute the pre-launch commands
  for (auto& command : impl_->pre_launch_commands_) { command->execute(workspace); }

  // Execute the launch commands
  for (auto& command : impl_->launch_commands_) { command->execute(workspace); }

  // Execute the post-launch commands
  for (auto& command : impl_->post_launch_commands_) { command->execute(workspace); }
}

}  // namespace holoscan::ops
