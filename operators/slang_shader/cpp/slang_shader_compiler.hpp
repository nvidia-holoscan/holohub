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

#ifndef SLANG_SHADER_COMPILER_HPP
#define SLANG_SHADER_COMPILER_HPP

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <slang-com-ptr.h>

#include "cuda_utils.hpp"

namespace holoscan::ops {

/**
 * @brief Manages Slang shader compilation and CUDA kernel retrieval
 *
 * The SlangShaderCompiler class provides functionality to compile Slang shader source code
 * into CUDA kernels that can be executed on GPU devices. It handles the complete
 * pipeline from shader compilation to kernel retrieval and parameter management.
 *
 * This class encapsulates the Slang compilation process, manages CUDA library
 * resources, and provides reflection information about the compiled shaders.
 * It supports global parameter updates for dynamic shader configuration.
 */
class SlangShaderCompiler {
 public:
  /**
   * @brief Constructs a SlangShaderCompiler with the given session and shader source
   *
   * @param session The Slang session used for compilation
   * @param shader_source The source code of the shader to compile
   *
   * This constructor compiles the provided shader source code using the Slang
   * compiler and prepares it for CUDA kernel retrieval. The compilation process
   * includes syntax checking, optimization, and code generation for CUDA targets.
   */
  explicit SlangShaderCompiler(const Slang::ComPtr<slang::ISession>& session,
                               const std::string& shader_source);

  /**
   * @brief Deleted default constructor
   *
   * SlangShaderCompiler requires a session and shader source for initialization,
   * so the default constructor is explicitly deleted.
   */
  SlangShaderCompiler() = delete;

  /**
   * @brief Retrieves reflection information about the compiled shader
   *
   * @return JSON object containing shader reflection data including:
   *         - Entry points and their signatures
   *         - Parameter layouts and types
   *         - Resource bindings and descriptors
   *         - Compilation metadata
   *
   * This method provides detailed information about the compiled shader's
   * structure, which is useful for understanding parameter layouts.
   */
  nlohmann::json get_reflection();

  /**
   * @brief Retrieves a CUDA kernel function by name
   *
   * @param name The name of the kernel function to retrieve
   * @return CUDA kernel handle that can be used for kernel execution
   *
   * This method looks up a compiled kernel function by its name and returns
   * a CUDA kernel handle. The kernel can then be launched using CUDA runtime
   * functions with appropriate grid and block dimensions.
   */
  cudaKernel_t get_kernel(const std::string& name);

  /**
   * @brief Updates global parameters for the shader
   *
   * @param name The name of the kernel to update the global parameters for
   * @param shader_parameters Binary data containing the new parameter values
   * @param stream CUDA stream to use for the parameter update
   *
   * This method updates the global parameters used by the shader. The parameters
   * are copied to GPU memory and will be used in subsequent kernel launches.
   * The size and layout of the parameters must match the shader's expected
   * parameter structure as defined in the shader source.
   */
  void update_global_params(const std::string& name, const std::vector<uint8_t>& shader_parameters,
                            cudaStream_t stream);

 private:
  /// Slang module containing the compiled shader code
  Slang::ComPtr<slang::IModule> module_;

  /// Linked program containing all shader entry points and resources
  Slang::ComPtr<slang::IComponentType> linked_program_;

  /// CUDA libraries containing the compiled kernels
  std::vector<UniqueCudaLibrary> cuda_libraries_;

  /// Size of the global parameters buffer in bytes
  size_t global_params_size_ = 0;

  /// Kernel information
  struct KernelInfo {
    void* dev_global_params_ = nullptr;   ///< Pointer to the device global parameters
    cudaKernel_t cuda_kernel_ = nullptr;  ///< CUDA kernel handle
  };

  /// Map of kernel names to kernel information
  std::map<std::string, KernelInfo> cuda_kernels_;
};

}  // namespace holoscan::ops

#endif /* SLANG_SHADER_COMPILER_HPP */
