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
#include "tensor_rt_inference.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>

// GXF 4.0 moved parameter_parser_std.hpp from std->core
#if __has_include("gxf/core/parameter_parser_std.hpp")
#include "gxf/core/parameter_parser_std.hpp"
#else
#include "gxf/std/parameter_parser_std.hpp"
#endif
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#define CUDA_TRY(stmt)                                                                     \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                    #stmt,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    cudaGetErrorString(_holoscan_cuda_err),                                \
                    _holoscan_cuda_err);                                                   \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })

namespace nvidia {
namespace holoscan {
namespace lstm_tensor_rt_inference {
namespace {

// Checks whether a string ends with a certain string
inline bool EndsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool IsValidFile(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) { return false; }
  return static_cast<bool>(st.st_mode & S_IFREG);
}

bool IsValidDirectory(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) { return false; }
  return static_cast<bool>(st.st_mode & S_IFDIR);
}

bool ReadEntireBinaryFile(const std::string& file_path, std::vector<char>& buffer) {
  // Make sure we are  opening a valid file.
  if (!IsValidFile(file_path)) { return false; }
  // Open the file in binary mode and seek to the end
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file) { return false; }
  // Get the size of the file and seek back to the beginning
  const size_t size = file.tellg();
  file.seekg(0);
  // Reserve enough space in the output buffer and read the file contents into it
  buffer.resize(size);
  const bool ret = static_cast<bool>(file.read(buffer.data(), size));
  file.close();
  return ret;
}

// Formats gxf tensor shape specified by std::array for console spew
const std::string FormatDims(const std::array<int32_t, gxf::Shape::kMaxRank>& dimensions,
                             const int32_t rank) {
  std::stringbuf sbuf;
  std::ostream stream(&sbuf);
  stream << "[";
  for (int i = 0; i < rank; ++i) {
    if (i > 0) { stream << ", "; }
    stream << dimensions[i];
  }
  stream << "]";
  return sbuf.str();
}

// Formats gxf shape for console spew
const std::string FormatTensorShape(const gxf::Shape& shape) {
  std::array<int32_t, gxf::Shape::kMaxRank> dimensions;
  for (uint32_t i = 0; i < shape.rank(); ++i) { dimensions[i] = shape.dimension(i); }
  return FormatDims(dimensions, shape.rank());
}

// Converts TensorRT dimensions to Gxf Tensor dimensions (std::array)
std::array<int32_t, gxf::Shape::kMaxRank> Dims2Dimensions(const nvinfer1::Dims& dims) {
  std::array<int32_t, gxf::Shape::kMaxRank> dimensions;
  dimensions.fill(1);
  for (int32_t i = 0; i < dims.nbDims; i++) { dimensions[i] = dims.d[i]; }
  return dimensions;
}

// Converts TensorRT data type to gxf::Tensor element type (gxf::PrimitiveType)
gxf::Expected<gxf::PrimitiveType> NvInferDatatypeToTensorElementType(nvinfer1::DataType data_type) {
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT: {
      return gxf::PrimitiveType::kFloat32;
    }
    case nvinfer1::DataType::kINT8: {
      return gxf::PrimitiveType::kInt8;
    }
    case nvinfer1::DataType::kINT32: {
      return gxf::PrimitiveType::kInt32;
    }
      //    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kHALF:
    default: {
      GXF_LOG_ERROR("Unsupported DataType %d", static_cast<int>(data_type));
      return gxf::Unexpected{GXF_FAILURE};
    }
  }
}

// Writes engine plan to specified file path
gxf::Expected<void> SerializeEnginePlan(const std::vector<char>& plan, const std::string& path) {
  // Write Plan To Disk
  std::ofstream out_stream(path.c_str(), std::ofstream::binary);
  if (!out_stream.is_open()) {
    GXF_LOG_ERROR("Failed to create engine file %s.", path.c_str());
    return gxf::Unexpected{GXF_FAILURE};
  }
  out_stream.write(plan.data(), plan.size());
  if (out_stream.bad()) {
    GXF_LOG_ERROR("Failed to writing to engine file %s.", path.c_str());
    return gxf::Unexpected{GXF_FAILURE};
  }
  out_stream.close();
  GXF_LOG_INFO("TensorRT engine serialized at %s", path.c_str());
  return gxf::Success;
}

}  // namespace

// Logging interface for the TensorRT builder, engine and runtime, to redirect logging,
void TensorRTInferenceLogger::log(ILogger::Severity severity, const char* msg) throw() {
  switch (severity) {
    case Severity::kINTERNAL_ERROR: {
      GXF_LOG_ERROR("TRT INTERNAL_ERROR: %s", msg);
      break;
    }
    case Severity::kERROR: {
      GXF_LOG_ERROR("TRT ERROR: %s", msg);
      break;
    }
    case Severity::kWARNING: {
      GXF_LOG_WARNING("TRT WARNING: %s", msg);
      break;
    }
    case Severity::kINFO: {
      GXF_LOG_DEBUG("TRT INFO: %s", msg);
      break;
    }
    case Severity::kVERBOSE: {
      if (verbose_) { GXF_LOG_DEBUG("TRT VERBOSE: %s", msg); }
      break;
    }
    default: {
      GXF_LOG_ERROR("TRT UNKNOWN SEVERITY ERROR: %s", msg);
      break;
    }
  }
}

void TensorRTInferenceLogger::setVerbose(bool verbose) {
  verbose_ = verbose;
}

gxf_result_t TensorRtInference::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      model_file_path_, "model_file_path", "Model File Path", "Path to ONNX model to be loaded.");
  result &= registrar->parameter(
      engine_cache_dir_,
      "engine_cache_dir",
      "Engine Cache Directory",
      "Path to a folder containing cached engine files to be serialized and loaded from.");
  result &= registrar->parameter(plugins_lib_namespace_,
                                 "plugins_lib_namespace",
                                 "Plugins Lib Namespace",
                                 "Namespace used to register all the plugins in this library.",
                                 std::string(""));
  result &= registrar->parameter(force_engine_update_,
                                 "force_engine_update",
                                 "Force Engine Update",
                                 "Always update engine regard less of existing engine file. "
                                 "Such conversion may take minutes. Default to false.",
                                 false);

  result &= registrar->parameter(input_tensor_names_,
                                 "input_tensor_names",
                                 "Input Tensor Names",
                                 "Names of input tensors in the order to be fed into the model.");
  result &= registrar->parameter(input_binding_names_,
                                 "input_binding_names",
                                 "Input Binding Names",
                                 "Names of input bindings as in the model in the same order of "
                                 "what is provided in input_tensor_names.");

  result &=
      registrar->parameter(input_state_tensor_names_,
                           "input_state_tensor_names",
                           "Input State Tensor Names",
                           "Names of input state tensors that are used internally by TensorRT.",
                           std::vector<std::string>{});

  result &= registrar->parameter(output_tensor_names_,
                                 "output_tensor_names",
                                 "Output Tensor Names",
                                 "Names of output tensors in the order to be retrieved "
                                 "from the model.");
  result &=
      registrar->parameter(output_state_tensor_names_,
                           "output_state_tensor_names",
                           "Output State Tensor Names",
                           "Names of output state tensors that are used internally by TensorRT.",
                           std::vector<std::string>{});

  result &= registrar->parameter(output_binding_names_,
                                 "output_binding_names",
                                 "Output Binding Names",
                                 "Names of output bindings in the model in the same "
                                 "order of of what is provided in output_tensor_names.");
  result &= registrar->parameter(pool_, "pool", "Pool", "Allocator instance for output tensors.");

  result &= registrar->parameter(max_workspace_size_,
                                 "max_workspace_size",
                                 "Max Workspace Size",
                                 "Size of working space in bytes. Default to 64MB",
                                 67108864l);
  result &= registrar->parameter(dla_core_,
                                 "dla_core",
                                 "DLA Core",
                                 "DLA Core to use. Fallback to GPU is always enabled. "
                                 "Default to use GPU only.",
                                 gxf::Registrar::NoDefaultParameter(),
                                 GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(max_batch_size_,
                                 "max_batch_size",
                                 "Max Batch Size",
                                 "Maximum possible batch size in case the first dimension is "
                                 "dynamic and used as batch size.",
                                 1);
  result &= registrar->parameter(enable_fp16_,
                                 "enable_fp16_",
                                 "Enable FP16 Mode",
                                 "Enable inference with FP16 and FP32 fallback.",
                                 false);

  result &= registrar->parameter(verbose_,
                                 "verbose",
                                 "Verbose",
                                 "Enable verbose logging on console. Default to false.",
                                 false);
  result &= registrar->parameter(relaxed_dimension_check_,
                                 "relaxed_dimension_check",
                                 "Relaxed Dimension Check",
                                 "Ignore dimensions of 1 for input tensor dimension check.",
                                 true);

  result &= registrar->parameter(rx_, "rx", "RX", "List of receivers to take input tensors");
  result &= registrar->parameter(tx_, "tx", "TX", "Transmitter to publish output tensors");

  result &= cuda_stream_handler_.registerInterface(registrar, true);

  return gxf::ToResultCode(result);
}

gxf_result_t TensorRtInference::start() {
  // Validates parameter
  if (!EndsWith(model_file_path_.get(), ".onnx")) {
    GXF_LOG_ERROR("Only supports ONNX model: %s.", model_file_path_.get().c_str());
    return GXF_FAILURE;
  }
  if (rx_.get().size() == 0) {
    GXF_LOG_ERROR("At least one receiver is needed.");
    return GXF_FAILURE;
  }

  if (input_tensor_names_.get().size() != input_binding_names_.get().size()) {
    GXF_LOG_ERROR("Mismatching number of input tensor names and bindings: %lu vs %lu.",
                  input_tensor_names_.get().size(),
                  input_binding_names_.get().size());
    return GXF_FAILURE;
  }
  if (output_tensor_names_.get().size() != output_binding_names_.get().size()) {
    GXF_LOG_ERROR("Mismatching number of output tensor names and bindings: %lu vs %lu.",
                  output_tensor_names_.get().size(),
                  output_binding_names_.get().size());
    return GXF_FAILURE;
  }

  // Check input and output state tensor names
  state_tensor_count_ = input_state_tensor_names_.get().size();
  if (input_state_tensor_names_.get().size() != output_state_tensor_names_.get().size()) {
    GXF_LOG_ERROR(
        "Number of output state tensors %d does not match number of input state "
        "tensors %d",
        static_cast<int>(output_state_tensor_names_.get().size()),
        static_cast<int>(input_state_tensor_names_.get().size()));
    return GXF_FAILURE;
  }

  // Initializes TensorRT registered plugins
  cuda_logger_.setVerbose(verbose_.get());
  const auto plugins_lib_namespace = plugins_lib_namespace_.try_get();
  if (plugins_lib_namespace &&
      !initLibNvInferPlugins(&cuda_logger_, plugins_lib_namespace.value().c_str())) {
    // Tries to proceed to see if the model would work
    GXF_LOG_WARNING("Could not initialize LibNvInferPlugins.");
  }

  gxf::Expected<std::string> maybe_host_engine_capability =
      queryHostEngineCapability(cuda_stream_handler_.getStreamHandle()->dev_id());
  if (!maybe_host_engine_capability) {
    GXF_LOG_ERROR("Failed to query host engine capability.");
    return GXF_FAILURE;
  }

  std::string host_engine_capability = maybe_host_engine_capability.value();
  GXF_LOG_INFO("Using Host Engine Capability: %s", host_engine_capability.c_str());

  gxf::Expected<std::string> maybe_engine_file_path = findEngineFilePath(host_engine_capability);
  if (!maybe_engine_file_path) {
    GXF_LOG_ERROR("Failed to find an engine file!");
    return GXF_FAILURE;
  }
  std::string engine_file_path = maybe_engine_file_path.value();
  engine_file_path_ = engine_file_path;

  if (force_engine_update_) {
    // Deletes engine plan file if exists for forced update
    std::remove(engine_file_path.c_str());
    if (std::ifstream(engine_file_path.c_str()).good()) {
      GXF_LOG_ERROR("Failed to remove engine plan file %s for forced engine update.",
                    engine_file_path.c_str());
      return GXF_FAILURE;
    }
  }

  // Loads Cuda engine into std::vector<char> plan or creates it if needed.
  std::vector<char> plan;
  if (force_engine_update_ || !ReadEntireBinaryFile(engine_file_path, plan)) {
    const char* warning_note = force_engine_update_ ? " (forced by config)" : "";
    GXF_LOG_WARNING(
        "Rebuilding CUDA engine %s%s. "
        "Note: this process may take up to several minutes.",
        engine_file_path.c_str(),
        warning_note);
    auto result = convertModelToEngine();
    if (!result) {
      GXF_LOG_ERROR("Failed to create engine plan for model %s.", model_file_path_.get().c_str());
      return gxf::ToResultCode(result);
    }

    // Skips loading file and uses in-memory engine plan directly.
    plan = std::move(result.value());

    // Tries to serializes the plan and proceeds anyway
    if (!SerializeEnginePlan(plan, engine_file_path)) {
      GXF_LOG_ERROR(
          "Engine plan serialization failed. Proceeds with in-memory engine plan anyway.");
    }
  }

  // Creates inference runtime for the plan
  infer_runtime_.reset(nvinfer1::createInferRuntime(cuda_logger_));

  // Deserialize the CUDA engine
  if (verbose_.get()) { GXF_LOG_DEBUG("Creating inference runtime."); }
  cuda_engine_.reset(infer_runtime_->deserializeCudaEngine(plan.data(), plan.size()));

  // Debug spews
  if (verbose_.get()) {
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    GXF_LOG_DEBUG("Number of CUDA bindings: %d", cuda_engine_->getNbBindings());
    for (int i = 0; i < cuda_engine_->getNbBindings(); ++i) {
      GXF_LOG_DEBUG("CUDA binding No.%d: name %s Format %s",
                    i,
                    cuda_engine_->getBindingName(i),
                    cuda_engine_->getBindingFormatDesc(i));
    }
#else
    GXF_LOG_DEBUG("Number of CUDA bindings: %d", cuda_engine_->getNbIOTensors());
    for (int i = 0; i < cuda_engine_->getNbIOTensors(); ++i) {
      GXF_LOG_DEBUG("CUDA binding No.%d: name %s Format %s",
                    i,
                    cuda_engine_->getIOTensorName(i),
                    cuda_engine_->getTensorFormatDesc(cuda_engine_->getIOTensorName(i)));
    }
#endif
  }

  // Checks binding numbers against parameter
  const uint64_t input_number = input_tensor_names_.get().size();
  const uint64_t output_number = output_tensor_names_.get().size();
  const int64_t total_bindings_number = input_number + output_number;
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
  if (cuda_engine_->getNbBindings() != static_cast<int>(total_bindings_number)) {
#else
  if (cuda_engine_->getNbIOTensors() != static_cast<int>(total_bindings_number)) {
#endif
    GXF_LOG_ERROR(
        "Numbers of CUDA bindings mismatch: configured for %lu vs model requires %d. "
        "Please check TensorRTInference codelet configuration.\n",
        total_bindings_number,
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
        cuda_engine_->getNbBindings());
#else
        cuda_engine_->getNbIOTensors());
#endif
    return GXF_ARGUMENT_INVALID;
  }

  // Creates cuda execution context
  cuda_execution_ctx_.reset(cuda_engine_->createExecutionContext());

  // Allocates CUDA buffer pointers for binding to be populated in tick()
  cuda_buffers_.resize(input_tensor_names_.get().size() + output_tensor_names_.get().size(),
                       nullptr);

  // Initialize internal state tensors
  internal_states_ = gxf::Entity::New(context());
  if (!internal_states_) { return gxf::ToResultCode(internal_states_); }

  // Keeps record of input bindings
  binding_infos_.clear();
  for (uint64_t j = 0; j < input_number; ++j) {
    const std::string& tensor_name = input_tensor_names_.get()[j];
    const std::string& binding_name = input_binding_names_.get()[j];

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    const int32_t binding_index = cuda_engine_->getBindingIndex(binding_name.c_str());
#else
    const int32_t binding_index = static_cast<int32_t>(j);
#endif
    if (binding_index == -1) {
      GXF_LOG_ERROR("Failed to get binding index for input %s in model %s",
                    binding_name.c_str(),
                    engine_file_path.c_str());
      return GXF_FAILURE;
    }

    if (binding_index >= static_cast<int>(cuda_buffers_.size())) {
      GXF_LOG_ERROR("Binding index for input %s is out of range in model %s.",
                    binding_name.c_str(),
                    engine_file_path.c_str());
      return GXF_FAILURE;
    }

    // Checks element type
    const auto maybe_element_type =
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
        NvInferDatatypeToTensorElementType(cuda_engine_->getBindingDataType(binding_index));
#else
        NvInferDatatypeToTensorElementType(cuda_engine_->getTensorDataType(binding_name.c_str()));
#endif
    if (!maybe_element_type) {
      GXF_LOG_ERROR("Unsupported element type for binding input %s on index %d. ",
                    binding_name.c_str(),
                    binding_index);
      return maybe_element_type.error();
    }

    // Keeps binding info
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    const auto& dims = cuda_engine_->getBindingDimensions(binding_index);
#else
    const auto& dims = cuda_engine_->getTensorShape(binding_name.c_str());
#endif
    binding_infos_[tensor_name] = BindingInfo{binding_index,
                                              static_cast<uint32_t>(dims.nbDims),
                                              binding_name,
                                              maybe_element_type.value(),
                                              Dims2Dimensions(dims)};

    // Debug spew
    if (verbose_.get()) {
      GXF_LOG_DEBUG(
          "Input Tensor %s:%s index %d Dimensions %s.",
          tensor_name.c_str(),
          binding_name.c_str(),
          binding_index,
          FormatDims(binding_infos_[tensor_name].dimensions, binding_infos_[tensor_name].rank)
              .c_str());
    }

    // Create tensor for input states
    if (std::find(input_state_tensor_names_.get().begin(),
                  input_state_tensor_names_.get().end(),
                  tensor_name) != input_state_tensor_names_.get().end()) {
      const BindingInfo& binding_info = binding_infos_[tensor_name];
      const gxf::Shape shape{Dims2Dimensions(dims), binding_info.rank};

      const auto maybe_input_state_tensor =
          internal_states_.value().add<gxf::Tensor>(tensor_name.c_str());
      if (!maybe_input_state_tensor) {
        GXF_LOG_ERROR("Failed to create input state tensor %s.", tensor_name.c_str());
        return maybe_input_state_tensor.error();
      }

      const auto result = maybe_input_state_tensor.value()->reshapeCustom(
          shape,
          binding_info.element_type,
          gxf::PrimitiveTypeSize(binding_info.element_type),
          gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
          gxf::MemoryStorageType::kDevice,
          pool_);
      if (!result) {
        GXF_LOG_ERROR("Failed to allocate for input state tensor %s", tensor_name.c_str());
        return gxf::ToResultCode(result);
      }
    }
  }

  // Keeps record of output bindings
  for (uint64_t j = 0; j < output_number; ++j) {
    const std::string& tensor_name = output_tensor_names_.get()[j];
    const std::string& binding_name = output_binding_names_.get()[j];

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    const int32_t binding_index = cuda_engine_->getBindingIndex(binding_name.c_str());
#else
    const int32_t binding_index = static_cast<int32_t>(input_number + j);
#endif
    if (binding_index == -1) {
      GXF_LOG_ERROR("Failed to get binding index for output %s", binding_name.c_str());
      return GXF_FAILURE;
    }
    if (binding_index >= static_cast<int>(cuda_buffers_.size())) {
      GXF_LOG_ERROR("Binding index for output %s is out of range. %d >= %d",
                    binding_name.c_str(),
                    binding_index,
                    static_cast<int>(cuda_buffers_.size()));
      return GXF_FAILURE;
    }

    // Checks element type
    const auto maybe_element_type =
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
      NvInferDatatypeToTensorElementType(cuda_engine_->getBindingDataType(binding_index));
#else
      NvInferDatatypeToTensorElementType(cuda_engine_->getTensorDataType(binding_name.c_str()));
#endif
     if (!maybe_element_type) {
      GXF_LOG_ERROR("Unsupported element type for binding output %s on index %d. ",
                    binding_name.c_str(),
                    binding_index);
      return maybe_element_type.error();
    }

    // Keeps binding info
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    const auto& dims = cuda_engine_->getBindingDimensions(binding_index);
#else
    const auto& dims = cuda_engine_->getTensorShape(binding_name.c_str());
#endif
    binding_infos_[tensor_name] = BindingInfo{binding_index,
                                              static_cast<uint32_t>(dims.nbDims),
                                              binding_name,
                                              maybe_element_type.value(),
                                              Dims2Dimensions(dims)};
    cuda_buffers_[binding_index] = nullptr;  // populate cuda_buffers dynamically, in tick()

    if (verbose_.get()) {
      GXF_LOG_DEBUG(
          "Output Tensor %s:%s (%d), Dimensions: %s.",
          tensor_name.c_str(),
          binding_name.c_str(),
          binding_index,
          FormatDims(binding_infos_[tensor_name].dimensions, binding_infos_[tensor_name].rank)
              .c_str());
    }
  }

  return GXF_SUCCESS;
}

gxf::Expected<std::vector<char>> TensorRtInference::convertModelToEngine() {
  // Creates the engine Builder
  NvInferHandle<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(cuda_logger_));

  // Builder Config provides options to the Builder
  NvInferHandle<nvinfer1::IBuilderConfig> builderConfig(builder->createBuilderConfig());
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
  builderConfig->setMaxWorkspaceSize(max_workspace_size_);
#else
  builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
#endif
  // Sets DLA core if provided and always fall back to GPU
  auto dla_core = dla_core_.try_get();
  if (dla_core) {
    builderConfig->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    builderConfig->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    builderConfig->setDLACore(dla_core.value());
  }
  if (enable_fp16_.get()) { builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16); }

  // Parses ONNX with explicit batch size for support of dynamic shapes/batch
  #if NV_TENSORRT_MAJOR < 10
    const auto explicitBatch =
              1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  #else
    const auto explicitBatch = 1U;
  #endif
  NvInferHandle<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));

  NvInferHandle<nvonnxparser::IParser> onnx_parser(
      nvonnxparser::createParser(*network, cuda_logger_));
  if (!onnx_parser->parseFromFile(model_file_path_.get().c_str(),
                                  static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    GXF_LOG_ERROR("Failed to parse ONNX file %s", model_file_path_.get().c_str());
    return gxf::Unexpected{GXF_FAILURE};
  }

  // Provides optimization profile for dynamic size input bindings
  nvinfer1::IOptimizationProfile* optimization_profile = builder->createOptimizationProfile();
  // Checks input dimensions and adds to optimization profile if needed
  const int number_inputs = network->getNbInputs();
  for (int i = 0; i < number_inputs; ++i) {
    auto* bind_tensor = network->getInput(i);
    const char* bind_name = bind_tensor->getName();
    nvinfer1::Dims dims = bind_tensor->getDimensions();

    // Validates binding info
    if (dims.nbDims <= 0) {
      GXF_LOG_ERROR("Invalid input tensor dimensions for binding %s", bind_name);
      return gxf::Unexpected{GXF_ARGUMENT_INVALID};
    }
    for (int j = 1; j < dims.nbDims; ++j) {
      if (dims.d[j] <= 0) {
        GXF_LOG_ERROR(
            "Input binding %s requires dynamic size on dimension No.%d which is not supported",
            bind_tensor->getName(),
            j);
        return gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
      }
    }
    if (dims.d[0] == -1) {
      // Only case with first dynamic dimension is supported and assumed to be batch size.
      // Always optimizes for 1-batch.
      dims.d[0] = 1;
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMIN, dims);
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kOPT, dims);
      dims.d[0] = max_batch_size_.get();
      if (max_batch_size_.get() <= 0) {
        GXF_LOG_ERROR("Maximum batch size %d is invalid. Uses 1 instead.", max_batch_size_.get());
        dims.d[0] = 1;
      }
      optimization_profile->setDimensions(bind_name, nvinfer1::OptProfileSelector::kMAX, dims);
    }
  }
  builderConfig->addOptimizationProfile(optimization_profile);

  // Creates TensorRT Engine Plan
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
  NvInferHandle<nvinfer1::IHostMemory> model_stream(builder->buildSerializedNetwork(
                                                    *network, *builderConfig));
#else
  NvInferHandle<nvinfer1::IHostMemory> model_stream(
      builder->buildSerializedNetwork(*network, *builderConfig));
#endif
  if (!model_stream || model_stream->size() == 0 || model_stream->data() == nullptr) {
    GXF_LOG_ERROR("Fail to serialize TensorRT Engine.");
    return gxf::Unexpected{GXF_FAILURE};
  }

  // Prepares return value
  std::vector<char> result;
  const char* data = static_cast<const char*>(model_stream->data());
  result.resize(model_stream->size());
  std::copy(data, data + model_stream->size(), result.data());
  return result;
}

gxf_result_t TensorRtInference::stop() {
  cuda_execution_ctx_ = nullptr;
  cuda_engine_ = nullptr;
  infer_runtime_ = nullptr;
  cuda_buffers_.clear();

  // Release resources for the internal states (tensor map as entity).
  // NOTE:: Without this, the following message can be seen at the end of the execution:
  //     "PANIC ./gxf/core/handle.hpp@174: Invalid Component Pointer."
  internal_states_ = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};

  return GXF_SUCCESS;
}

gxf_result_t TensorRtInference::tick() {
  // Grabs latest messages from all receivers
  std::vector<gxf::Entity> messages;
  messages.reserve(rx_.get().size());
  for (auto& rx : rx_.get()) {
    gxf::Expected<gxf::Entity> maybe_message = rx->receive();
    if (maybe_message) { messages.push_back(std::move(maybe_message.value())); }
  }
  if (messages.empty()) {
    GXF_LOG_ERROR("No message available.");
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }

  // sync with the CUDA streams from the input messages
  gxf_result_t stream_handler_result = cuda_stream_handler_.fromMessages(context(), messages);
  if (stream_handler_result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to get the CUDA stream from incoming messages");
    return stream_handler_result;
  }
  // Tries to retrieve timestamp if clock present
  gxf::Expected<gxf::Handle<gxf::Timestamp>> maybe_input_timestamp = gxf::Unexpected{GXF_FAILURE};
  for (auto& msg : messages) {
    maybe_input_timestamp = msg.get<gxf::Timestamp>("timestamp");
    if (maybe_input_timestamp) { break; }
  }
  // Populates input tensors
  for (uint32_t input_index = 0; input_index < input_tensor_names_.get().size(); ++input_index) {
    const auto& tensor_name = input_tensor_names_.get()[input_index];
    const std::string& binding_name = input_binding_names_.get()[input_index];

    gxf::Expected<gxf::Handle<gxf::Tensor>> maybe_tensor = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
    if (std::find(input_state_tensor_names_.get().begin(),
                  input_state_tensor_names_.get().end(),
                  tensor_name) == input_state_tensor_names_.get().end()) {
      for (auto& msg : messages) {
        maybe_tensor = msg.get<gxf::Tensor>(tensor_name.c_str());
        if (maybe_tensor) { break; }
      }
      if (!maybe_tensor) {
        GXF_LOG_ERROR("Failed to retrieve Tensor %s", tensor_name.c_str());
        return GXF_FAILURE;
      }
    } else {
      maybe_tensor = internal_states_.value().get<gxf::Tensor>(tensor_name.c_str());
      if (!maybe_tensor) {
        GXF_LOG_ERROR("Failed to retrieve Tensor %s from internal states", tensor_name.c_str());
        return GXF_FAILURE;
      }
    }

    if (!maybe_tensor) {
      GXF_LOG_ERROR("Failed to retrieve Tensor %s", tensor_name.c_str());
      return GXF_FAILURE;
    }

    gxf::Tensor& input_tensor = *maybe_tensor.value();

    // Validates input tensor against model bindings then binds and populates buffers
    const auto& shape = input_tensor.shape();
    const auto& binding_info = binding_infos_[tensor_name];
    nvinfer1::Dims dims;
    dims.nbDims = binding_info.rank;
    for (int32_t i = 0; i < dims.nbDims; ++i) { dims.d[i] = binding_info.dimensions[i]; }

    // Checks input tensor element type
    if (input_tensor.element_type() != binding_info.element_type) {
      GXF_LOG_ERROR("Mismatching tensor element type required %d vs provided %d",
                    static_cast<int>(binding_info.element_type),
                    static_cast<int>(input_tensor.element_type()));
      return GXF_FAILURE;
    }

    if (relaxed_dimension_check_.get()) {
      // Relaxed dimension match. Ignore all 1s. Binding of -1 is considered as match.
      const uint32_t shape_rank = shape.rank();
      uint32_t shape_rank_matched = 0;
      uint32_t binding_rank_matched = 0;
      bool matched = true;
      for (uint32_t i = 0; i < gxf::Shape::kMaxRank * 2; ++i) {
        if (shape_rank_matched >= shape_rank || binding_rank_matched >= binding_info.rank) {
          break;
        }
        if (shape.dimension(shape_rank_matched) == 1) {
          shape_rank_matched++;
          continue;
        }
        if (binding_info.dimensions[binding_rank_matched] == 1) {
          binding_rank_matched++;
          continue;
        }
        if (binding_info.dimensions[binding_rank_matched] == -1) {
          // Matches dimension
          dims.d[binding_rank_matched] = shape.dimension(shape_rank_matched);
          shape_rank_matched++;
          binding_rank_matched++;
          continue;
        }
        if (shape.dimension(shape_rank_matched) != binding_info.dimensions[binding_rank_matched]) {
          matched = false;
          break;
        }
        shape_rank_matched++;
        binding_rank_matched++;
      }
      if (!matched || shape_rank_matched != shape_rank ||
          binding_rank_matched != binding_info.rank) {
        GXF_LOG_ERROR(
            "Input Tensor %s bound to %s:"
            " dimensions does not meet model spec with relaxed matching. Expected: %s Real: %s",
            tensor_name.c_str(),
            binding_info.binding_name.c_str(),
            FormatDims(binding_info.dimensions, binding_info.rank).c_str(),
            FormatTensorShape(shape).c_str());
        return GXF_FAILURE;
      }
    } else {
      // Strict dimension match. All dimensions must match. Binding of -1 is considered as match.
      if (shape.rank() != binding_info.rank) {
        GXF_LOG_ERROR("Tensor %s bound to %s has mismatching rank %d (%d required)",
                      tensor_name.c_str(),
                      binding_info.binding_name.c_str(),
                      shape.rank(),
                      binding_info.rank);
        return GXF_FAILURE;
      }
      for (uint32_t i = 0; i < binding_info.rank; i++) {
        if (binding_info.dimensions[i] == -1) { dims.d[i] = shape.dimension(i); }
        if (shape.dimension(i) != binding_info.dimensions[i] && binding_info.dimensions[i] != -1) {
          GXF_LOG_ERROR("Tensor %s bound to %s has mismatching dimension %d:%d (%d required)",
                        tensor_name.c_str(),
                        binding_info.binding_name.c_str(),
                        i,
                        shape.dimension(i),
                        binding_info.dimensions[i]);
          return GXF_FAILURE;
        }
      }
    }

    // Updates the latest dimension of input tensor
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    if (!cuda_execution_ctx_->setBindingDimensions(binding_info.index, dims)) {
#else
    if (!cuda_execution_ctx_->setInputShape(binding_name.c_str(), dims)) {
#endif
      GXF_LOG_ERROR("Failed to update input binding %s dimensions.",
                    binding_info.binding_name.c_str());
      return GXF_FAILURE;
    }

    // Binds input tensor buffer
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    cuda_buffers_[binding_info.index] = input_tensor.pointer();
#else
    cuda_execution_ctx_->setTensorAddress(binding_name.c_str(), input_tensor.pointer());
#endif
  }

  // Creates result message entity
  gxf::Expected<gxf::Entity> maybe_result_message = gxf::Entity::New(context());
  if (!maybe_result_message) { return gxf::ToResultCode(maybe_result_message); }

  // Creates tensors for output
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
  for (const auto& tensor_name : output_tensor_names_.get()) {
#else
  for (uint32_t output_index = 0; output_index < output_tensor_names_.get().size();
       ++output_index) {
    const auto& tensor_name = output_tensor_names_.get()[output_index];
    const std::string& binding_name = output_binding_names_.get()[output_index];
#endif
    auto maybe_result_tensor = maybe_result_message.value().add<gxf::Tensor>(tensor_name.c_str());
    if (!maybe_result_tensor) {
      GXF_LOG_ERROR("Failed to create output tensor %s", tensor_name.c_str());
      return gxf::ToResultCode(maybe_result_tensor);
    }

    // Queries binding dimension from context and allocates tensor accordingly
    const auto& binding_info = binding_infos_[tensor_name];
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    const auto binding_dims = cuda_engine_->getBindingDimensions(binding_info.index);
#else
    const auto binding_dims = cuda_engine_->getTensorShape(binding_name.c_str());
#endif
    gxf::Shape shape{Dims2Dimensions(binding_dims), binding_info.rank};

    auto result = maybe_result_tensor.value()->reshapeCustom(
        shape,
        binding_info.element_type,
        gxf::PrimitiveTypeSize(binding_info.element_type),
        gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
        gxf::MemoryStorageType::kDevice,
        pool_);
    if (!result) {
      GXF_LOG_ERROR("Failed to allocate for output tensor %s", tensor_name.c_str());
      return gxf::ToResultCode(result);
    }

    // Allocates gpu buffer for output tensors
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
    cuda_buffers_[binding_info.index] = maybe_result_tensor.value()->pointer();
#else
    cuda_execution_ctx_->setTensorAddress(binding_name.c_str(),
                                          maybe_result_tensor.value()->pointer());
#endif
  }

  // Runs inference on specified CUDA stream
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR <5)
  if (!cuda_execution_ctx_->enqueueV2(
          cuda_buffers_.data(), cuda_stream_handler_.getCudaStream(), nullptr)) {
#else
  if (!cuda_execution_ctx_->enqueueV3(cuda_stream_handler_.getCudaStream())) {
#endif
    GXF_LOG_ERROR("TensorRT task enqueue for engine %s failed.", engine_file_path_.c_str());
    return GXF_FAILURE;
  }
  // Copy output state tensor to the input state tensor of the next stage
  for (uint32_t i = 0; i < state_tensor_count_; ++i) {
    // Get output state tensor
    const auto out_state_tensor = maybe_result_message.value()
                                      .get<gxf::Tensor>(output_state_tensor_names_.get()[i].c_str())
                                      .value();
    if (!out_state_tensor) {
      GXF_LOG_ERROR("Failed to get output tensor %s", output_state_tensor_names_.get()[i].c_str());
      return GXF_FAILURE;
    }
    const auto out_tensor_ptr = out_state_tensor->pointer();
    if (!out_tensor_ptr) {
      GXF_LOG_ERROR("Output tensor %s has null pointer",
                    output_state_tensor_names_.get()[i].c_str());
      return GXF_FAILURE;
    }

    // Get input state tensor
    const auto in_state_tensor = internal_states_.value()
                                     .get<gxf::Tensor>(input_state_tensor_names_.get()[i].c_str())
                                     .value();
    if (!in_state_tensor) {
      GXF_LOG_ERROR("Failed to get output tensor %s", output_state_tensor_names_.get()[i].c_str());
      return GXF_FAILURE;
    }
    const auto in_state_tensor_ptr = in_state_tensor->pointer();
    if (!in_state_tensor_ptr) {
      GXF_LOG_ERROR("Input state tensor %s has null pointer",
                    input_state_tensor_names_.get()[i].c_str());
      return GXF_FAILURE;
    }

    const size_t state_tensor_size = out_state_tensor->size();
    if (state_tensor_size != in_state_tensor->size()) {
      GXF_LOG_ERROR(
          "Output state tensor %s (size: %d) does not match input state "
          "tensor %s (size: %d)",
          output_state_tensor_names_.get()[i].c_str(),
          static_cast<int>(state_tensor_size),
          input_state_tensor_names_.get()[i].c_str(),
          static_cast<int>(in_state_tensor->size()));
      return GXF_FAILURE;
    }

    // Copy output state tensor to input state tensor
    cudaError_t cuda_status = CUDA_TRY(cudaMemcpyAsync(in_state_tensor_ptr,
                                                       out_tensor_ptr,
                                                       state_tensor_size,
                                                       cudaMemcpyDeviceToDevice,
                                                       cuda_stream_handler_.getCudaStream()));
    if (cuda_status) {
      GXF_LOG_ERROR("Failed to copy output tensor %s to input tensor %s: %s",
                    output_state_tensor_names_.get()[0].c_str(),
                    input_state_tensor_names_.get()[0].c_str(),
                    cudaGetErrorString(cuda_status));
      return GXF_FAILURE;
    }
  }

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.toMessage(maybe_result_message);
  if (stream_handler_result != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to add the CUDA stream to the outgoing messages");
    return stream_handler_result;
  }

  // Publishes result with acqtime
  if (maybe_input_timestamp) {  // if input timestamp is present, use it's acqtime
    return gxf::ToResultCode(
        tx_->publish(maybe_result_message.value(), maybe_input_timestamp.value()->acqtime));
  } else {  // else simply use 0 as acqtime
    return gxf::ToResultCode(tx_->publish(maybe_result_message.value(), 0));
  }
}

static std::string replaceChar(const std::string& string, char match, char replacement) {
  std::string result = string;
  std::replace(result.begin(), result.end(), match, replacement);
  return result;
}

gxf::Expected<std::string> TensorRtInference::queryHostEngineCapability(int dev_id) const {
  char* env_var = std::getenv("GXF_TENSORRT_HOST_ENGINE_CAPABILITY");
  if (env_var != nullptr) {
    GXF_LOG_INFO("Using GXF_TENSORRT_HOST_ENGINE_CAPABILITY overwrite: %s", env_var);
    return std::string(env_var);
  }
  cudaDeviceProp device_prop = {0};
  cudaError_t status = cudaGetDeviceProperties(&device_prop, dev_id);
  if (status != cudaSuccess) {
    GXF_LOG_ERROR("Failed to get cuda device properties with errorcode: %d", status);
    return gxf::Unexpected{};
  }
  std::string device_name = device_prop.name;
  device_name = replaceChar(device_name, ' ', '-');
  std::stringstream ss;
  // TensorRT builds an engine file per device that changes based on the number of SMs available.
  // This returns a string that should be a unique mapping per device and SM configuration.
  ss << device_name << "_c" << device_prop.major << device_prop.minor << "_n"
     << device_prop.multiProcessorCount;
  return ss.str();
}

gxf::Expected<std::string> TensorRtInference::findEngineFilePath(
    const std::string& host_engine_capability) const {
  std::string engine_file_path;
  if (!IsValidDirectory(engine_cache_dir_.get())) {
    // Create the directory
    if (!std::filesystem::is_directory(engine_cache_dir_.get()) ||
        !std::filesystem::exists(engine_cache_dir_.get())) {  // Check if src folder exists
      if (!std::filesystem::create_directory(engine_cache_dir_.get())) {
        GXF_LOG_ERROR(
            "Cannot create engine cache directory '%s'! Please create a valid cache directory.",
            engine_cache_dir_.get().c_str());
        return gxf::Unexpected{};
      }
    }
  }
  engine_file_path = engine_cache_dir_.get() + "/" + host_engine_capability + ".engine";
  GXF_LOG_INFO("Loading engine cache dir file: %s", engine_file_path.c_str());
  if (engine_file_path.empty()) {
    GXF_LOG_ERROR("Engine file path not specified!");
    return gxf::Unexpected{};
  }
  return engine_file_path;
}

}  // namespace lstm_tensor_rt_inference
}  // namespace holoscan
}  // namespace nvidia
