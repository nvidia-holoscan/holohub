/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TAPNEXT_INFERENCE_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TAPNEXT_INFERENCE_HPP_

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

#include "../utils/cuda_stream_handler.hpp"

namespace nvidia::holoscan::tapnext_inference {

// Logger for TensorRT to redirect logging into gxf console spew.
class TensorRTInferenceLogger : public nvinfer1::ILogger {
 public:
  void log(ILogger::Severity severity, const char* msg) throw() override;
  // Sets verbose flag for logging
  void setVerbose(bool verbose);

 private:
  bool verbose_;
};

/// @brief Loads ONNX models (Init and Fwd), takes input tensors and run inference.
///
/// This codelet is specifically designed for TapNext-like architectures where an initialization
/// model runs on the first step, and a forward model runs on subsequent steps, maintaining
/// internal state between steps.
class TapNextInference : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  // Helper to return a string for the TRT engine capability.
  gxf::Expected<std::string> queryHostEngineCapability(int dev_id) const;
  // Helper to search for the engine file path.
  gxf::Expected<std::string> findEngineFilePath(const std::string& host_engine_capability, const std::string& model_path) const;
  // Helper to convert model to engine
  gxf::Expected<std::vector<char>> convertModelToEngine(const std::string& model_path, int32_t max_batch_size);

  // Helper deleter to call destroy while destroying the cuda objects
  template <typename T>
  struct DeleteFunctor {
    inline void operator()(void* ptr) { delete reinterpret_cast<T*>(ptr); }
  };
  // unique_ptr using custom Delete Functor above
  template <typename T>
  using NvInferHandle = std::unique_ptr<T, DeleteFunctor<T>>;

  // To cache binding info for tensors
  typedef struct {
    int32_t index;
    uint32_t rank;
    std::string binding_name;
    gxf::PrimitiveType element_type;
    std::array<int32_t, gxf::Shape::kMaxRank> dimensions;
  } BindingInfo;

  struct EngineContext {
    std::string model_path;
    std::string engine_file_path;
    NvInferHandle<nvinfer1::IExecutionContext> cuda_execution_ctx;
    NvInferHandle<nvinfer1::ICudaEngine> cuda_engine;
    NvInferHandle<nvinfer1::IRuntime> infer_runtime;
    std::vector<void*> cuda_buffers;
    std::unordered_map<std::string, BindingInfo> binding_infos;

    // Config
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> input_binding_names;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_binding_names;
  };

  gxf_result_t setupEngine(EngineContext& ctx,
                           const std::string& model_path,
                           const std::vector<std::string>& in_tensors,
                           const std::vector<std::string>& in_bindings,
                           const std::vector<std::string>& out_tensors,
                           const std::vector<std::string>& out_bindings);

  // Helper to create query points
  gxf_result_t createQueryPoints(int32_t batch_size);

  gxf::Parameter<std::string> model_file_path_init_;
  gxf::Parameter<std::string> model_file_path_fwd_;
  gxf::Parameter<std::string> engine_cache_dir_;
  gxf::Parameter<std::string> plugins_lib_namespace_;
  gxf::Parameter<bool> force_engine_update_;

  // Init Model bindings
  gxf::Parameter<std::vector<std::string>> input_tensor_names_init_;
  gxf::Parameter<std::vector<std::string>> input_binding_names_init_;
  gxf::Parameter<std::vector<std::string>> output_tensor_names_init_;
  gxf::Parameter<std::vector<std::string>> output_binding_names_init_;

  // Fwd Model bindings
  gxf::Parameter<std::vector<std::string>> input_tensor_names_fwd_;
  gxf::Parameter<std::vector<std::string>> input_binding_names_fwd_;
  gxf::Parameter<std::vector<std::string>> output_tensor_names_fwd_;
  gxf::Parameter<std::vector<std::string>> output_binding_names_fwd_;

  // State tensors (shared between init output, fwd input, fwd output)
  gxf::Parameter<std::vector<std::string>> state_tensor_names_;

  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<int64_t> max_workspace_size_;
  gxf::Parameter<int32_t> max_batch_size_;
  gxf::Parameter<bool> enable_fp16_;
  gxf::Parameter<bool> relaxed_dimension_check_;
  gxf::Parameter<bool> verbose_;

  // Grid generation params
  gxf::Parameter<int32_t> grid_size_;
  gxf::Parameter<int32_t> grid_height_;
  gxf::Parameter<int32_t> grid_width_;

  gxf::Parameter<std::vector<gxf::Handle<gxf::Receiver>>> rx_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> tx_;

  // Logger instance for TensorRT
  TensorRTInferenceLogger cuda_logger_;

  EngineContext init_engine_ctx_;
  EngineContext fwd_engine_ctx_;

  // Internal state storage
  gxf::Expected<gxf::Entity> internal_states_ = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
  // Stored query points
  gxf::Expected<gxf::Entity> static_states_ = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};

  holoscan::CudaStreamHandler cuda_stream_handler_;
};

}  // namespace nvidia::holoscan::tapnext_inference

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TAPNEXT_INFERENCE_HPP_
