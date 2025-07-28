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

#include "include/slang_shader/slang_shader.hpp"

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/core/resources/gxf/rmm_allocator.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "command.hpp"
#include "slang_shader_compiler.hpp"
#include "slang_utils.hpp"

#include "holoscan_slang.hpp"

/**
 * Custom YAML parser for PreprocessorMacros class
 */
template <>
struct YAML::convert<holoscan::ops::SlangShaderOp::PreprocessorMacros> {
  static Node encode(const holoscan::ops::SlangShaderOp::PreprocessorMacros& preprocessor_macros) {
    Node node;
    for (const auto& [key, value] : preprocessor_macros) { node[key] = value; }
    return node;
  }

  static bool decode(const Node& node,
                     holoscan::ops::SlangShaderOp::PreprocessorMacros& preprocessor_macros) {
    if (!node.IsMap()) {
      HOLOSCAN_LOG_ERROR("InputSpec: expected a map");
      return false;
    }
    try {
      for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        preprocessor_macros[it->first.as<std::string>()] = it->second.as<std::string>();
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(e.what());
      return false;
    }
    return true;
  }
};

namespace holoscan::ops {

// Implementation struct containing all Slang-related details
class SlangShaderOp::Impl {
 public:
  Impl() {
    // First we need to create slang global session to work with the Slang API.
    SLANG_CALL(slang::createGlobalSession(global_session_.writeRef()));
  }

  /**
   * @brief Create a Slang session
   *
   * @param preprocessor_macros Preprocessor macros to be used in the session
   */
  void create_session(const std::map<std::string, std::string>& preprocessor_macros) {
    // Create Session
    slang::SessionDesc sessionDesc;

    // Set up target description for PTX
    slang::TargetDesc targetDesc;
    targetDesc.format = SLANG_PTX;

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    std::vector<slang::PreprocessorMacroDesc> preprocessor_macros_desc;

    for (auto& [name, value] : preprocessor_macros) {
      slang::PreprocessorMacroDesc preprocessor_macro_desc;
      preprocessor_macro_desc.name = name.c_str();
      preprocessor_macro_desc.value = value.c_str();
      preprocessor_macros_desc.push_back(preprocessor_macro_desc);
    }

    sessionDesc.preprocessorMacroCount = preprocessor_macros_desc.size();
    sessionDesc.preprocessorMacros = preprocessor_macros_desc.data();

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

  std::unique_ptr<SlangShaderCompiler> shader_compiler_;

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
 * @param parameter The JSON reflection data containing parameter type and binding information
 * @param param_name The name of the parameter to create
 * @return A unique_ptr to the created CommandParameter, or nullptr if the type is unsupported
 */
std::unique_ptr<CommandParameter> create_command_parameter(OperatorSpec& spec,
                                                           const nlohmann::json& parameter,
                                                           const std::string& param_name) {
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

  register_converter<PreprocessorMacros>();

  spec.param(shader_source_, "shader_source", "Shader source string.", "Shader source string.");

  spec.param(
      shader_source_file_, "shader_source_file", "Shader source file.", "Shader source file.");

  spec.param(preprocessor_macros_,
             "preprocessor_macros",
             "Preprocessor macros to be used when compiling the shader.",
             "The map consists of string pairs, where the key is the macro name and the value is "
             "the macro value.",
             {});

  spec.param(
      allocator_,
      "allocator",
      "Allocator for output buffers.",
      "Allocator for output buffers.",
      std::static_pointer_cast<Allocator>(fragment()->make_resource<RMMAllocator>("allocator")));

  // Add a CUDA stream pool
  add_arg(fragment()->make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 0));

  // We need the shader source and preprocessor macros to build the input and output ports and the
  // parameters, so check the argument list and get them
  std::string shader_source, shader_source_file;
  std::map<std::string, std::string> preprocessor_macros;
  for (auto&& arg : args()) {
    if (arg.name() == "shader_source") {
      shader_source = std::any_cast<std::string>(arg.value());
    } else if (arg.name() == "shader_source_file") {
      shader_source_file = std::any_cast<std::string>(arg.value());
    } else if (arg.name() == "preprocessor_macros") {
      preprocessor_macros = std::any_cast<std::map<std::string, std::string>>(arg.value());
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

  impl_->create_session(preprocessor_macros);

  impl_->shader_compiler_ = std::make_unique<SlangShaderCompiler>(impl_->session_, shader_source);

  const nlohmann::json reflection = impl_->shader_compiler_->get_reflection();

  // Get Slang parameters and setup inputs, outputs and parameters
  for (auto& parameter : reflection["parameters"]) {
    if (!parameter.contains("userAttribs")) {
      continue;
    }

    // We need to store the output command for the alloc_size_of or alloc command
    CommandOutput* command_output = nullptr;

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
        auto [input_port_name, input_item_name] = split(input_name, ':');
        // If the input port is not defined, define it
        if (spec.inputs().find(input_port_name) == spec.inputs().end()) {
          spec.input<gxf::Entity>(input_port_name);
        }
        impl_->pre_launch_commands_.push_back(std::make_unique<CommandInput>(
            input_port_name, input_item_name, parameter["name"], parameter["binding"]["offset"]));
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
        auto [output_port_name, output_item_name] = split(output_name, ':');
        // If the output port is not defined, define it
        if (spec.outputs().find(output_port_name) == spec.outputs().end()) {
          spec.output<gxf::Entity>(output_port_name);
        }

        auto command =
            std::make_unique<CommandOutput>(output_port_name, output_item_name, parameter["name"]);
        // Keep the output command for the alloc_size_of command
        command_output = command.get();
        impl_->post_launch_commands_.push_back(std::move(command));
      } else if ((attrib_name == "alloc_size_of") || (attrib_name == "alloc")) {
        // size_of
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }

        if (!command_output) {
          throw std::runtime_error(
              fmt::format("Attribute '{}' requires an output attribute to be "
                          "defined before it.",
                          attrib_name));
        }

        std::string reference_name;
        uint32_t size_x = 0;
        uint32_t size_y = 0;
        uint32_t size_z = 0;

        if (attrib_name == "alloc_size_of") {
          reference_name = user_attrib["arguments"].at(0);
        } else {
          size_x = user_attrib["arguments"].at(0);
          size_y = user_attrib["arguments"].at(1);
          size_z = user_attrib["arguments"].at(2);
        }

        std::string element_type;
        uint32_t element_count;
        if (parameter["type"]["resultType"]["kind"] == "vector") {
          element_type =
              parameter["type"]["resultType"]["elementType"]["scalarType"].get<std::string>();
          element_count = parameter["type"]["resultType"]["elementCount"];
        } else if (parameter["type"]["resultType"]["kind"] == "scalar") {
          element_type = parameter["type"]["resultType"]["scalarType"].get<std::string>();
          element_count = 1;
        } else {
          throw std::runtime_error(
              fmt::format("Attribute '{}' supports vectors and scalars only "
                          "for the result type.",
                          attrib_name));
        }
        impl_->pre_launch_commands_.push_back(
            std::make_unique<CommandAlloc>(command_output->port_name(),
                                           command_output->item_name(),
                                           parameter["name"],
                                           reference_name,
                                           size_x,
                                           size_y,
                                           size_z,
                                           element_type,
                                           element_count,
                                           allocator_,
                                           parameter["binding"]["offset"]));
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
        auto command_parameter = create_command_parameter(spec, parameter, param_name);
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
            (parameter["type"]["elementType"]["scalarType"] != "uint32")) {
          throw std::runtime_error(
              fmt::format("Attribute '{}' supports a three component uint32 vector (`uint3`) "
                          "uniforms only and cannot be applied to '{}'.",
                          attrib_name,
                          parameter["name"].get<std::string>()));
        }
        const std::string size_of_name = user_attrib["arguments"].at(0);
        impl_->pre_launch_commands_.push_back(std::make_unique<CommandSizeOf>(
            parameter["name"], size_of_name, parameter["binding"]["offset"]));
      } else if (attrib_name == "strides_of") {
        // strides_of
        if ((parameter["binding"]["kind"] != "uniform") ||
            (parameter["type"]["kind"] != "vector") || (parameter["type"]["elementCount"] != 3) ||
            (parameter["type"]["elementType"]["kind"] != "scalar") ||
            (parameter["type"]["elementType"]["scalarType"] != "uint64")) {
          throw std::runtime_error(
              fmt::format("Attribute '{}' supports a three component uint64 vector (`uint64_t3`) "
                          "uniforms only and cannot be applied to '{}'.",
                          attrib_name,
                          parameter["name"].get<std::string>()));
        }
        const std::string strides_of_name = user_attrib["arguments"].at(0);
        impl_->pre_launch_commands_.push_back(std::make_unique<CommandStrideOf>(
            parameter["name"], strides_of_name, parameter["binding"]["offset"]));
      } else if (attrib_name == "zeros") {
        // zeros
        // input
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }

        impl_->pre_launch_commands_.push_back(std::make_unique<CommandZeros>(parameter["name"]));
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

    std::string invocations_size_of_name;
    dim3 invocations(1, 1, 1);

    // Check the user attributes and check for Holoscan attributes
    if (entry_point.contains("userAttribs")) {
      for (auto& user_attrib : entry_point["userAttribs"]) {
        const std::string user_attrib_name = user_attrib["name"];
        const std::string holoscan_prefix = "holoscan_";
        if (user_attrib_name.find(holoscan_prefix) != 0) {
          continue;
        }

        const std::string attrib_name = user_attrib_name.substr(holoscan_prefix.size());

        if (attrib_name == "invocations_size_of") {
          if (user_attrib["arguments"].empty()) {
            throw std::runtime_error("Attribute 'invocations_size_of' requires an argument");
          }
          invocations_size_of_name = user_attrib["arguments"].at(0);
        } else if (attrib_name == "invocations") {
          if (user_attrib["arguments"].empty()) {
            throw std::runtime_error("Attribute 'invocations' requires an argument");
          }
          invocations.x = user_attrib["arguments"].at(0);
          if (user_attrib["arguments"].size() > 1) {
            invocations.y = user_attrib["arguments"].at(1);
            if (user_attrib["arguments"].size() > 2) {
              invocations.z = user_attrib["arguments"].at(2);
            }
          }
        } else {
          throw std::runtime_error("Unknown user attribute: " + user_attrib_name);
        }
      }
    }

    const dim3 thread_group_size = dim3(entry_point["threadGroupSize"][0],
                                        entry_point["threadGroupSize"][1],
                                        entry_point["threadGroupSize"][2]);

    // And then launch the kernel
    impl_->launch_commands_.push_back(std::make_unique<CommandLaunch>(entry_point["name"],
                                                                      impl_->shader_compiler_.get(),
                                                                      thread_group_size,
                                                                      invocations_size_of_name,
                                                                      invocations));
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
