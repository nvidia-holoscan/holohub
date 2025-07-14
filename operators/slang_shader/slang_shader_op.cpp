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

    /// @todo Add support for preprocessor macros

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
 * @brief Converts a string to lowercase
 *
 * @param s The string to convert
 * @return The lowercase string
 */
std::string to_lower(const std::string& s) {
  std::string s_lower = s;
  std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return s_lower;
}

/**
 * @brief Converts a string to a specific type
 *
 * @tparam typeT The type to convert to
 * @param s The string to convert
 * @return The converted value
 */
template <typename typeT>
typeT from_string(const std::string& s);

/**
 * @brief Converts a string to a boolean
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
bool from_string(const std::string& s) {
  const std::string s_lower = to_lower(s);
  if (s_lower == "true") {
    return true;
  } else if (s_lower == "false") {
    return false;
  } else {
    throw std::runtime_error(fmt::format("Invalid boolean value: {}", s));
  }
}

/**
 * @brief Converts a string to an int8_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int8_t from_string(const std::string& s) {
  return static_cast<int8_t>(std::stoi(s));
}

/**
 * @brief Converts a string to a uint8_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint8_t from_string(const std::string& s) {
  return static_cast<uint8_t>(std::stoul(s));
}

/**
 * @brief Converts a string to an int16_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int16_t from_string(const std::string& s) {
  return static_cast<int16_t>(std::stoi(s));
}

/**
 * @brief Converts a string to a uint16_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint16_t from_string(const std::string& s) {
  return static_cast<uint16_t>(std::stoul(s));
}

/**
 * @brief Converts a string to an int32_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int32_t from_string(const std::string& s) {
  return static_cast<int32_t>(std::stoi(s));
}

/**
 * @brief Converts a string to a uint32_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint32_t from_string(const std::string& s) {
  return static_cast<uint32_t>(std::stoul(s));
}

/**
 * @brief Converts a string to an int64_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int64_t from_string(const std::string& s) {
  return static_cast<int64_t>(std::stoll(s));
}

/**
 * @brief Converts a string to a uint64_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint64_t from_string(const std::string& s) {
  return static_cast<uint64_t>(std::stoull(s));
}

/**
 * @brief Converts a string to a float
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
float from_string(const std::string& s) {
  return static_cast<float>(std::stof(s));
}

/**
 * @brief Converts a string to a double
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
double from_string(const std::string& s) {
  return static_cast<double>(std::stod(s));
}
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
 * @param default_value The default value of the parameter
 * @return A unique_ptr to the created CommandParameter, or nullptr if the type is unsupported
 */
std::unique_ptr<CommandParameter> create_command_parameter(OperatorSpec& spec,
                                                           const nlohmann::json& parameter,
                                                           const std::string& param_name,
                                                           const std::string& default_value) {
  if (parameter["type"]["scalarType"] == "bool") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<bool>(),
                                              param_name,
                                              from_string<bool>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int32") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<int32_t>(),
                                              param_name,
                                              from_string<int32_t>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint32") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<uint32_t>(),
                                              param_name,
                                              from_string<uint32_t>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int64") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<int64_t>(),
                                              param_name,
                                              from_string<int64_t>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint64") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<uint64_t>(),
                                              param_name,
                                              from_string<uint64_t>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "float32") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<float>(),
                                              param_name,
                                              from_string<float>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "float64") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<double>(),
                                              param_name,
                                              from_string<double>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int8") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<int8_t>(),
                                              param_name,
                                              from_string<int8_t>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint8") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<uint8_t>(),
                                              param_name,
                                              from_string<uint8_t>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "int16") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<int16_t>(),
                                              param_name,
                                              from_string<int16_t>(default_value),
                                              parameter["binding"]["offset"]);
  } else if (parameter["type"]["scalarType"] == "uint16") {
    return std::make_unique<CommandParameter>(spec,
                                              new Parameter<uint16_t>(),
                                              param_name,
                                              from_string<uint16_t>(default_value),
                                              parameter["binding"]["offset"]);
  }

  return nullptr;
}

/**
 * Splits a string into a pair of strings, separated by a colon.
 *
 * @param s The string to split
 * @return A pair of strings, the first is the part before the colon, the second is the part after
 */
std::pair<std::string, std::string> split(const std::string& s) {
  auto colon_pos = s.find(':');
  if (colon_pos != std::string::npos) {
    return {s.substr(0, colon_pos), s.substr(colon_pos + 1)};
  } else {
    return {s, ""};
  }
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

    // We need to store the output command for the alloc_size_of command
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
        auto [input_port_name, input_item_name] = split(input_name);
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
        auto [output_port_name, output_item_name] = split(output_name);
        // If the output port is not defined, define it
        if (spec.outputs().find(output_port_name) == spec.outputs().end()) {
          spec.output<gxf::Entity>(output_port_name);
        }

        auto command =
            std::make_unique<CommandOutput>(output_port_name, output_item_name, parameter["name"]);
        // Keep the output command for the alloc_size_of command
        command_output = command.get();
        impl_->post_launch_commands_.push_back(std::move(command));
      } else if (attrib_name == "alloc_size_of") {
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

        const std::string reference_name = user_attrib["arguments"].at(0);
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
            std::make_unique<CommandAllocSizeOf>(command_output->port_name(),
                                                 command_output->item_name(),
                                                 parameter["name"],
                                                 reference_name,
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
        const std::string default_value = user_attrib["arguments"].at(1);
        auto command_parameter =
            create_command_parameter(spec, parameter, param_name, default_value);
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

    dim3 block_size(1, 1, 1);
    if (entry_point.contains("threadGroupSize")) {
      block_size.x = entry_point["threadGroupSize"][0];
      block_size.y = entry_point["threadGroupSize"][1];
      block_size.z = entry_point["threadGroupSize"][2];
    }

    // And then launch the kernel
    impl_->launch_commands_.push_back(std::make_unique<CommandLaunch>(entry_point["name"],
                                                                      impl_->shader_.get(),
                                                                      block_size,
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
