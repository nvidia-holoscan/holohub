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

#include "slang_shader_compiler.hpp"

#include <driver_types.h>
#include <array>

#include "slang_utils.hpp"

namespace holoscan::ops {

SlangShaderCompiler::SlangShaderCompiler(const Slang::ComPtr<slang::ISession>& session,
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

  // Create composite component type (module + entry point)
  std::vector<slang::IComponentType*> component_types{module_.get()};
  Slang::ComPtr<slang::IEntryPoint> entry_points[entry_point_count];

  // Add entry points to the component types
  for (int i = 0; i < entry_point_count; ++i) {
    SLANG_CALL(module_->getDefinedEntryPoint(i, entry_points[i].writeRef()));
    component_types.push_back(entry_points[i].get());
  }

  Slang::ComPtr<slang::IComponentType> composed_program;
  SLANG_CALL_WITH_DIAGNOSTICS(session->createCompositeComponentType(component_types.data(),
                                                                    component_types.size(),
                                                                    composed_program.writeRef(),
                                                                    diagnostics_blob.writeRef()));

  // Link the program
  SLANG_CALL_WITH_DIAGNOSTICS(
      composed_program->link(linked_program_.writeRef(), diagnostics_blob.writeRef()));

  // Generate target code
  for (int i = 0; i < entry_point_count; ++i) {
    Slang::ComPtr<ISlangBlob> ptx_code;
    SLANG_CALL_WITH_DIAGNOSTICS(linked_program_->getEntryPointCode(i,  // entryPointIndex
                                                                   0,  // targetIndex
                                                                   ptx_code.writeRef(),
                                                                   diagnostics_blob.writeRef()));
    // Uncomment this to print the PTX code
    // HOLOSCAN_LOG_INFO("PTX code: {}", (const char*)ptx_code->getBufferPointer());

    cuda_libraries_.emplace_back([&ptx_code] {
      cudaLibrary_t library;
      CUDA_CALL(cudaLibraryLoadData(
          &library, ptx_code->getBufferPointer(), nullptr, nullptr, 0, nullptr, nullptr, 0));
      return library;
    }());

    KernelInfo kernel_info;

    // Get the pointer to the global params memory
    size_t global_params_size;
    cudaError result = cudaLibraryGetGlobal(&kernel_info.dev_global_params_,
                                            &global_params_size,
                                            cuda_libraries_.back().get(),
                                            "SLANG_globalParams");
    if (result == cudaSuccess) {
      if (i == 0) {
        global_params_size_ = global_params_size;
      } else if (global_params_size != global_params_size_) {
        throw std::runtime_error(fmt::format("Global params size mismatch, expected: {}, got: {}",
                                             global_params_size_,
                                             global_params_size));
      }
    } else if (result != cudaErrorSymbolNotFound) {
      // Global params are optional, we ignore this error but report any other errors
      CUDA_CALL(result);
    }

    unsigned int kernel_count;
    CUDA_CALL(cudaLibraryGetKernelCount(&kernel_count, cuda_libraries_.back().get()));
    if (kernel_count != 1) {
      throw std::runtime_error(fmt::format("Expected 1 kernel, got {}", kernel_count));
    }

    slang::FunctionReflection* function_reflection;
    function_reflection = entry_points[i]->getFunctionReflection();

    CUDA_CALL(
        cudaLibraryEnumerateKernels(&kernel_info.cuda_kernel_, 1, cuda_libraries_.back().get()));
    cuda_kernels_[std::string(function_reflection->getName())] = kernel_info;
  }
}

nlohmann::json SlangShaderCompiler::get_reflection() {
  // Get reflection from the linked program and convert to JSON
  Slang::ComPtr<ISlangBlob> reflection_blob;
  SLANG_CALL(linked_program_->getLayout(0)->toJson(reflection_blob.writeRef()));
  // Uncomment this to print the reflection
  // HOLOSCAN_LOG_INFO("Reflection: {}", (const char*)reflection_blob->getBufferPointer());

  return nlohmann::json::parse((const char*)reflection_blob->getBufferPointer());
}

cudaKernel_t SlangShaderCompiler::get_kernel(const std::string& name) {
  auto it = cuda_kernels_.find(name);
  if (it == cuda_kernels_.end()) {
    throw std::runtime_error(fmt::format("Kernel '{}' not found", name));
  }

  return it->second.cuda_kernel_;
}

void SlangShaderCompiler::update_global_params(const std::string& name,
                                               const std::vector<uint8_t>& shader_parameters,
                                               cudaStream_t stream) {
  auto it = cuda_kernels_.find(name);
  if (it == cuda_kernels_.end()) {
    throw std::runtime_error(fmt::format("Kernel '{}' not found", name));
  }

  if (shader_parameters.size() > global_params_size_) {
    throw std::runtime_error(fmt::format(
        "Shader parameters size mismatch, shader_parameters.size(): {}, global_params_size_: {}",
        shader_parameters.size(),
        global_params_size_));
  }

  // Copy the shader parameters to the device (note: this is done unconditionally since the GPU
  // addresses of the resources contained here are expected to change on every call)
  CUDA_CALL(cudaMemcpyAsync(it->second.dev_global_params_,
                            shader_parameters.data(),
                            shader_parameters.size(),
                            cudaMemcpyHostToDevice,
                            stream));
}

}  // namespace holoscan::ops
