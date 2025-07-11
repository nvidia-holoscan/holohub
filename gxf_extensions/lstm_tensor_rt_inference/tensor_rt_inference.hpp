/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_LSTM_TENSOR_RT_INFERENCE_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_LSTM_TENSOR_RT_INFERENCE_HPP_

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

namespace nvidia::holoscan::lstm_tensor_rt_inference {

// Logger for TensorRT to redirect logging into gxf console spew.
class TensorRTInferenceLogger : public nvinfer1::ILogger {
 public:
  void log(ILogger::Severity severity, const char* msg) throw() override;
  // Sets verbose flag for logging
  void setVerbose(bool verbose);

 private:
  bool verbose_;
};

/// @brief Loads ONNX model, takes input tensors and run inference against them with TensorRT.
///
/// It takes input from all receivers provided and try to locate Tensor component with specified
/// name on them one by one. The first occurrence would be used. Only takes gpu memory tensor.
/// Supports dynamic batch as first dimension.
/// The codelet has an engine cache directory that can be pre-populated to reduce start time.
/// If the engine cache directory has no pre-existing engine file for an architecture, it will
/// generate this dynamically.
/// Requires gxf::CudaStream to run load on specific CUDA stream.
class TensorRtInference : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  // Helper to return a string for the TRT engine capability.
  gxf::Expected<std::string> queryHostEngineCapability(int dev_id) const;
  // Helper to search for the engine file path.
  gxf::Expected<std::string> findEngineFilePath(const std::string& host_engine_capability) const;

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
  std::unordered_map<std::string, BindingInfo> binding_infos_;

  // Converts loaded model to engine plan
  gxf::Expected<std::vector<char>> convertModelToEngine();

  gxf::Parameter<std::string> model_file_path_;
  gxf::Parameter<std::string> engine_cache_dir_;
  gxf::Parameter<std::string> plugins_lib_namespace_;
  gxf::Parameter<bool> force_engine_update_;
  gxf::Parameter<std::vector<std::string>> input_tensor_names_;
  gxf::Parameter<std::vector<std::string>> input_state_tensor_names_;
  gxf::Parameter<std::vector<std::string>> input_binding_names_;
  gxf::Parameter<std::vector<std::string>> output_tensor_names_;
  gxf::Parameter<std::vector<std::string>> output_state_tensor_names_;
  gxf::Parameter<std::vector<std::string>> output_binding_names_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  gxf::Parameter<int64_t> max_workspace_size_;
  gxf::Parameter<int32_t> dla_core_;
  gxf::Parameter<int32_t> max_batch_size_;
  gxf::Parameter<bool> enable_fp16_;
  gxf::Parameter<bool> relaxed_dimension_check_;
  gxf::Parameter<bool> verbose_;

  gxf::Parameter<std::vector<gxf::Handle<gxf::Receiver>>> rx_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> tx_;

  // Logger instance for TensorRT
  TensorRTInferenceLogger cuda_logger_;

  NvInferHandle<nvinfer1::IExecutionContext> cuda_execution_ctx_;
  NvInferHandle<nvinfer1::ICudaEngine> cuda_engine_;
  NvInferHandle<nvinfer1::IRuntime> infer_runtime_;

  std::vector<void*> cuda_buffers_;

  uint32_t state_tensor_count_ = 0;
  gxf::Expected<gxf::Entity> internal_states_ = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
  std::string engine_file_path_;

  holoscan::CudaStreamHandler cuda_stream_handler_;
};

}  // namespace nvidia::holoscan::lstm_tensor_rt_inference

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_LSTM_TENSOR_RT_INFERENCE_HPP_
