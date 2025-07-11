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

#include "slang_shader.hpp"

#include <array>

#include "slang_utils.hpp"

namespace holoscan::ops {

SlangShader::SlangShader(const Slang::ComPtr<slang::ISession>& session,
                         const std::string& shader_source) {
  // Load module from source string
  Slang::ComPtr<ISlangBlob> diagnostics_blob;

  module_ = session->loadModuleFromSourceString(
      "user", "user.slang", shader_source.c_str(), diagnostics_blob.writeRef());
  // Check for compilation errors
  SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);
  if (!module_) {
    throw std::runtime_error("Failed to compile Slang module");
  }

  // Find entry points in the module
  const int entry_point_count = module_->getDefinedEntryPointCount();
  if (entry_point_count == 0) {
    throw std::runtime_error("Warning: No entry points found in shader");
  }

  // Get the first entry point
  Slang::ComPtr<slang::IEntryPoint> entryPoint;
  SLANG_CALL(module_->getDefinedEntryPoint(0, entryPoint.writeRef()));

  // Create composite component type (module + entry point)
  std::array<slang::IComponentType*, 2> component_types = {module_, entryPoint};
  Slang::ComPtr<slang::IComponentType> composed_program;

  SLANG_CALL_WITH_DIAGNOSTICS(session->createCompositeComponentType(component_types.data(),
                                                                    component_types.size(),
                                                                    composed_program.writeRef(),
                                                                    diagnostics_blob.writeRef()));

  // Link the program
  SLANG_CALL_WITH_DIAGNOSTICS(
      composed_program->link(linked_program_.writeRef(), diagnostics_blob.writeRef()));

  // Generate target code
  Slang::ComPtr<ISlangBlob> ptx_code;
  SLANG_CALL_WITH_DIAGNOSTICS(linked_program_->getEntryPointCode(0,  // entryPointIndex
                                                                 0,  // targetIndex
                                                                 ptx_code.writeRef(),
                                                                 diagnostics_blob.writeRef()));
  // Uncomment this to print the PTX code
  // HOLOSCAN_LOG_INFO("PTX code: {}", (const char*)ptx_code->getBufferPointer());

  cuda_library_.reset([&ptx_code] {
    cudaLibrary_t library;
    CUDA_CALL(cudaLibraryLoadData(
        &library, ptx_code->getBufferPointer(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    return library;
  }());

  // Get the pointer to the global params memory
  CUDA_CALL(cudaLibraryGetGlobal(
      &dev_global_params_, &global_params_size_, cuda_library_.get(), "SLANG_globalParams"));
}

nlohmann::json SlangShader::get_reflection() {
  // Get reflection from the linked program and convert to JSON
  Slang::ComPtr<ISlangBlob> reflection_blob;
  SLANG_CALL(linked_program_->getLayout(0)->toJson(reflection_blob.writeRef()));
  // Uncomment this to print the reflection
  // HOLOSCAN_LOG_INFO("Reflection: {}", (const char*)reflection_blob->getBufferPointer());

  return nlohmann::json::parse((const char*)reflection_blob->getBufferPointer());
}

cudaKernel_t SlangShader::get_kernel(const std::string& name) {
  cudaKernel_t kernel;
  CUDA_CALL(cudaLibraryGetKernel(&kernel, cuda_library_.get(), name.c_str()));
  return kernel;
}

void SlangShader::update_global_params(const std::vector<uint8_t>& shader_parameters) {
  if (shader_parameters.size() > global_params_size_) {
    throw std::runtime_error(fmt::format(
        "Shader parameters size mismatch, shader_parameters.size(): {}, global_params_size_: {}",
        shader_parameters.size(),
        global_params_size_));
  }

  // Copy the host global params to the device
  CUDA_CALL(cudaMemcpy(dev_global_params_,
                       shader_parameters.data(),
                       shader_parameters.size(),
                       cudaMemcpyHostToDevice));
}

}  // namespace holoscan::ops
