/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#include "tapnext_inference.hpp"

#include <NvInferPlugin.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
namespace tapnext_inference {
namespace {

// Checks whether a string ends with a certain string
inline bool EndsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool IsValidFile(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return false;
  }
  return static_cast<bool>(st.st_mode & S_IFREG);
}

bool IsValidDirectory(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return false;
  }
  return static_cast<bool>(st.st_mode & S_IFDIR);
}

bool ReadEntireBinaryFile(const std::string& file_path, std::vector<char>& buffer) {
  if (!IsValidFile(file_path)) {
    return false;
  }
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  const size_t size = file.tellg();
  file.seekg(0);
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
    if (i > 0) {
      stream << ", ";
    }
    stream << dimensions[i];
  }
  stream << "]";
  return sbuf.str();
}

const std::string FormatTensorShape(const gxf::Shape& shape) {
  std::array<int32_t, gxf::Shape::kMaxRank> dimensions;
  for (uint32_t i = 0; i < shape.rank(); ++i) { dimensions[i] = shape.dimension(i); }
  return FormatDims(dimensions, shape.rank());
}

std::array<int32_t, gxf::Shape::kMaxRank> Dims2Dimensions(const nvinfer1::Dims& dims) {
  std::array<int32_t, gxf::Shape::kMaxRank> dimensions;
  dimensions.fill(1);
  for (int32_t i = 0; i < dims.nbDims; i++) { dimensions[i] = dims.d[i]; }
  return dimensions;
}

gxf::Expected<gxf::PrimitiveType> NvInferDatatypeToTensorElementType(nvinfer1::DataType data_type) {
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      return gxf::PrimitiveType::kFloat32;
    case nvinfer1::DataType::kINT8:
      return gxf::PrimitiveType::kInt8;
    case nvinfer1::DataType::kINT32:
      return gxf::PrimitiveType::kInt32;
    case nvinfer1::DataType::kHALF:
      return gxf::PrimitiveType::kFloat16;
    default: {
      GXF_LOG_ERROR("Unsupported DataType %d", static_cast<int>(data_type));
      return gxf::Unexpected{GXF_FAILURE};
    }
  }
}

gxf::Expected<void> SerializeEnginePlan(const std::vector<char>& plan, const std::string& path) {
  std::ofstream out_stream(path.c_str(), std::ofstream::binary);
  if (!out_stream.is_open()) {
    GXF_LOG_ERROR("Failed to create engine file %s.", path.c_str());
    return gxf::Unexpected{GXF_FAILURE};
  }
  out_stream.write(plan.data(), plan.size());
  if (out_stream.bad()) {
    GXF_LOG_ERROR("Failed to write to engine file %s.", path.c_str());
    return gxf::Unexpected{GXF_FAILURE};
  }
  out_stream.close();
  GXF_LOG_INFO("TensorRT engine serialized at %s", path.c_str());
  return gxf::Success;
}

static std::string replaceChar(const std::string& string, char match, char replacement) {
  std::string result = string;
  std::replace(result.begin(), result.end(), match, replacement);
  return result;
}

}  // namespace

void TensorRTInferenceLogger::log(ILogger::Severity severity, const char* msg) throw() {
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      GXF_LOG_ERROR("TRT INTERNAL_ERROR: %s", msg);
      break;
    case Severity::kERROR:
      GXF_LOG_ERROR("TRT ERROR: %s", msg);
      break;
    case Severity::kWARNING:
      GXF_LOG_WARNING("TRT WARNING: %s", msg);
      break;
    case Severity::kINFO:
      GXF_LOG_DEBUG("TRT INFO: %s", msg);
      break;
    case Severity::kVERBOSE:
      if (verbose_) {
        GXF_LOG_DEBUG("TRT VERBOSE: %s", msg);
      }
      break;
    default:
      GXF_LOG_ERROR("TRT UNKNOWN SEVERITY ERROR: %s", msg);
      break;
  }
}

void TensorRTInferenceLogger::setVerbose(bool verbose) {
  verbose_ = verbose;
}

gxf_result_t TapNextInference::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(model_file_path_init_,
                                 "model_file_path_init",
                                 "Init Model File Path",
                                 "Path to ONNX model for initialization.");
  result &= registrar->parameter(model_file_path_fwd_,
                                 "model_file_path_fwd",
                                 "Forward Model File Path",
                                 "Path to ONNX model for forward tracking.");

  result &= registrar->parameter(engine_cache_dir_,
                                 "engine_cache_dir",
                                 "Engine Cache Directory",
                                 "Path to folder for cached engine files.");
  result &= registrar->parameter(plugins_lib_namespace_,
                                 "plugins_lib_namespace",
                                 "Plugins Lib Namespace",
                                 "",
                                 std::string(""));
  result &= registrar->parameter(
      force_engine_update_, "force_engine_update", "Force Engine Update", "", false);

  result &= registrar->parameter(input_tensor_names_init_,
                                 "input_tensor_names_init",
                                 "Init Input Tensor Names",
                                 "Input tensors for Init model.");
  result &= registrar->parameter(input_binding_names_init_,
                                 "input_binding_names_init",
                                 "Init Input Binding Names",
                                 "Input bindings for Init model.");
  result &= registrar->parameter(output_tensor_names_init_,
                                 "output_tensor_names_init",
                                 "Init Output Tensor Names",
                                 "Output tensors for Init model.");
  result &= registrar->parameter(output_binding_names_init_,
                                 "output_binding_names_init",
                                 "Init Output Binding Names",
                                 "Output bindings for Init model.");

  result &= registrar->parameter(input_tensor_names_fwd_,
                                 "input_tensor_names_fwd",
                                 "Fwd Input Tensor Names",
                                 "Input tensors for Fwd model.");
  result &= registrar->parameter(input_binding_names_fwd_,
                                 "input_binding_names_fwd",
                                 "Fwd Input Binding Names",
                                 "Input bindings for Fwd model.");
  result &= registrar->parameter(output_tensor_names_fwd_,
                                 "output_tensor_names_fwd",
                                 "Fwd Output Tensor Names",
                                 "Output tensors for Fwd model.");
  result &= registrar->parameter(output_binding_names_fwd_,
                                 "output_binding_names_fwd",
                                 "Fwd Output Binding Names",
                                 "Output bindings for Fwd model.");

  result &= registrar->parameter(
      state_tensor_names_,
      "state_tensor_names",
      "State Tensor Names",
      "List of tensor names that are treated as internal states (preserved across steps).");

  result &= registrar->parameter(pool_, "pool", "Pool", "Allocator instance.");

  result &= registrar->parameter(
      max_workspace_size_, "max_workspace_size", "Max Workspace Size", "", 67108864l);
  result &= registrar->parameter(max_batch_size_, "max_batch_size", "Max Batch Size", "", 1);
  result &= registrar->parameter(enable_fp16_, "enable_fp16", "Enable FP16", "", false);
  result &= registrar->parameter(verbose_, "verbose", "Verbose", "", false);
  result &= registrar->parameter(
      relaxed_dimension_check_, "relaxed_dimension_check", "Relaxed Dimension Check", "", true);

  result &=
      registrar->parameter(grid_size_, "grid_size", "Grid Size", "Grid size for query points", 15);
  result &= registrar->parameter(
      grid_height_, "grid_height", "Grid Height", "Image height for grid generation", 256);
  result &= registrar->parameter(
      grid_width_, "grid_width", "Grid Width", "Image width for grid generation", 256);

  result &= registrar->parameter(rx_, "rx", "RX", "List of receivers");
  result &= registrar->parameter(tx_, "tx", "TX", "Transmitter");

  result &= cuda_stream_handler_.registerInterface(registrar, true);

  return gxf::ToResultCode(result);
}

gxf_result_t TapNextInference::start() {
  cuda_logger_.setVerbose(verbose_.get());
  const auto plugins_lib_namespace = plugins_lib_namespace_.try_get();
  if (plugins_lib_namespace &&
      !initLibNvInferPlugins(&cuda_logger_, plugins_lib_namespace.value().c_str())) {
    GXF_LOG_WARNING("Could not initialize LibNvInferPlugins.");
  }

  // Initialize Internal State Entity
  internal_states_ = gxf::Entity::New(context());
  if (!internal_states_)
    return gxf::ToResultCode(internal_states_);

  // Initialize Static State Entity (for query points)
  static_states_ = gxf::Entity::New(context());
  if (!static_states_)
    return gxf::ToResultCode(static_states_);

  // Setup Init Engine
  gxf_result_t res = setupEngine(init_engine_ctx_,
                                 model_file_path_init_.get(),
                                 input_tensor_names_init_.get(),
                                 input_binding_names_init_.get(),
                                 output_tensor_names_init_.get(),
                                 output_binding_names_init_.get());
  if (res != GXF_SUCCESS)
    return res;

  GXF_LOG_INFO("Init Engine Setup Done");

  // Setup Fwd Engine
  res = setupEngine(fwd_engine_ctx_,
                    model_file_path_fwd_.get(),
                    input_tensor_names_fwd_.get(),
                    input_binding_names_fwd_.get(),
                    output_tensor_names_fwd_.get(),
                    output_binding_names_fwd_.get());
  if (res != GXF_SUCCESS)
    return res;

  GXF_LOG_INFO("Fwd Engine Setup Done");

  // Create Query Points
  res = createQueryPoints(max_batch_size_.get());
  if (res != GXF_SUCCESS)
    return res;

  GXF_LOG_INFO("Query Points Created");

  return GXF_SUCCESS;
}

gxf_result_t TapNextInference::setupEngine(EngineContext& ctx, const std::string& model_path,
                                           const std::vector<std::string>& in_tensors,
                                           const std::vector<std::string>& in_bindings,
                                           const std::vector<std::string>& out_tensors,
                                           const std::vector<std::string>& out_bindings) {
  ctx.model_path = model_path;
  ctx.input_tensor_names = in_tensors;
  ctx.input_binding_names = in_bindings;
  ctx.output_tensor_names = out_tensors;
  ctx.output_binding_names = out_bindings;

  if (EndsWith(model_path, ".engine") || EndsWith(model_path, ".plan")) {
    ctx.engine_file_path = model_path;
    // Skip cache/conversion logic
  } else {
    if (!EndsWith(model_path, ".onnx")) {
      GXF_LOG_ERROR("Supports ONNX (.onnx) or Engine (.engine/.plan) models: %s",
                    model_path.c_str());
      return GXF_FAILURE;
    }

    gxf::Expected<std::string> maybe_host_engine_capability =
        queryHostEngineCapability(cuda_stream_handler_.getStreamHandle()->dev_id());
    if (!maybe_host_engine_capability)
      return GXF_FAILURE;

    std::string host_engine_capability = maybe_host_engine_capability.value();
    gxf::Expected<std::string> maybe_engine_file_path =
        findEngineFilePath(host_engine_capability, model_path);
    if (!maybe_engine_file_path)
      return GXF_FAILURE;

    ctx.engine_file_path = maybe_engine_file_path.value();

    const bool force_engine_update = force_engine_update_.get();

    if (force_engine_update) {
      std::remove(ctx.engine_file_path.c_str());
    }

    std::vector<char> plan;
    if (force_engine_update || !ReadEntireBinaryFile(ctx.engine_file_path, plan)) {
      GXF_LOG_WARNING("Rebuilding CUDA engine %s. This may take a while.",
                      ctx.engine_file_path.c_str());
      auto result = convertModelToEngine(model_path, max_batch_size_.get());
      if (!result)
        return gxf::ToResultCode(result);
      plan = std::move(result.value());
      auto serialize_result = SerializeEnginePlan(plan, ctx.engine_file_path);
      if (!serialize_result) {
        GXF_LOG_ERROR("Failed to serialize engine plan to file: %s", ctx.engine_file_path.c_str());
        return gxf::ToResultCode(serialize_result);
      }
    }
  }

  std::vector<char> plan;
  if (!ReadEntireBinaryFile(ctx.engine_file_path, plan)) {
    GXF_LOG_ERROR("Failed to read engine file: %s", ctx.engine_file_path.c_str());
    return GXF_FAILURE;
  }

  ctx.infer_runtime.reset(nvinfer1::createInferRuntime(cuda_logger_));
  ctx.cuda_engine.reset(ctx.infer_runtime->deserializeCudaEngine(plan.data(), plan.size()));

  if (!ctx.cuda_engine) {
    GXF_LOG_ERROR("Failed to deserialize engine: %s", ctx.engine_file_path.c_str());
    return GXF_FAILURE;
  }

  ctx.cuda_execution_ctx.reset(ctx.cuda_engine->createExecutionContext());

  // Allocate buffer pointers
  ctx.cuda_buffers.resize(in_tensors.size() + out_tensors.size(), nullptr);
  ctx.binding_infos.clear();

  // Populate binding infos
  const int n_inputs = in_tensors.size();
  for (int i = 0; i < n_inputs + (int)out_tensors.size(); ++i) {
    bool is_input = i < n_inputs;
    std::string tensor_name = is_input ? in_tensors[i] : out_tensors[i - n_inputs];
    std::string binding_name = is_input ? in_bindings[i] : out_bindings[i - n_inputs];

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
    int32_t binding_index = ctx.cuda_engine->getBindingIndex(binding_name.c_str());
#else
    int32_t binding_index = i;  // EnqueueV3 uses flat indices usually, but we need names mostly.
#endif

    // We will use names for lookup in V3.
    // BindingInfo is used for validation.

    nvinfer1::DataType dtype;
    nvinfer1::Dims dims;

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
    dtype = ctx.cuda_engine->getBindingDataType(binding_index);
    dims = ctx.cuda_engine->getBindingDimensions(binding_index);
#else
    dtype = ctx.cuda_engine->getTensorDataType(binding_name.c_str());
    dims = ctx.cuda_engine->getTensorShape(binding_name.c_str());
    // binding_index is not really used for V3 except as an ID we assign
    binding_index = i;
#endif

    auto elem_type_exp = NvInferDatatypeToTensorElementType(dtype);
    if (!elem_type_exp)
      return GXF_FAILURE;

    ctx.binding_infos[tensor_name] = BindingInfo{binding_index,
                                                 static_cast<uint32_t>(dims.nbDims),
                                                 binding_name,
                                                 elem_type_exp.value(),
                                                 Dims2Dimensions(dims)};

    // Initialize state tensors in internal_states_ if they appear here
    // Check if this tensor is a state tensor
    bool is_state = false;
    for (const auto& s : state_tensor_names_.get()) {
      if (s == tensor_name) {
        is_state = true;
        break;
      }
    }

    if (is_state && is_input) {
      // Allocate input state tensor
      auto maybe_state_tensor = internal_states_.value().add<gxf::Tensor>(tensor_name.c_str());
      if (!maybe_state_tensor)
        return maybe_state_tensor.error();

      gxf::Shape shape{Dims2Dimensions(dims), static_cast<uint32_t>(dims.nbDims)};
      auto res =
          maybe_state_tensor.value()->reshapeCustom(shape,
                                                    elem_type_exp.value(),
                                                    gxf::PrimitiveTypeSize(elem_type_exp.value()),
                                                    gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                                    gxf::MemoryStorageType::kDevice,
                                                    pool_);
      if (!res)
        return gxf::ToResultCode(res);
      // Zero init
      cudaMemset(maybe_state_tensor.value()->pointer(), 0, maybe_state_tensor.value()->size());
    }
  }

  return GXF_SUCCESS;
}

gxf_result_t TapNextInference::stop() {
  init_engine_ctx_.cuda_execution_ctx.reset();
  init_engine_ctx_.cuda_engine.reset();
  init_engine_ctx_.infer_runtime.reset();

  fwd_engine_ctx_.cuda_execution_ctx.reset();
  fwd_engine_ctx_.cuda_engine.reset();
  fwd_engine_ctx_.infer_runtime.reset();

  internal_states_ = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
  static_states_ = gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
  return GXF_SUCCESS;
}

gxf_result_t TapNextInference::tick() {
  // 1. Receive Messages
  std::vector<gxf::Entity> messages;
  for (auto& rx : rx_.get()) {
    auto msg = rx->receive();
    if (msg)
      messages.push_back(std::move(msg.value()));
  }
  if (messages.empty())
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;

  if (cuda_stream_handler_.fromMessages(context(), messages) != GXF_SUCCESS)
    return GXF_FAILURE;

  // Find frame and step
  gxf::Handle<gxf::Tensor> step_tensor;
  gxf::Handle<gxf::Tensor> frame_tensor;
  gxf::Handle<gxf::Timestamp> input_timestamp;

  for (auto& msg : messages) {
    auto maybe_step = msg.get<gxf::Tensor>("step");
    if (maybe_step)
      step_tensor = maybe_step.value();

    auto maybe_frame = msg.get<gxf::Tensor>("frame");
    if (maybe_frame)
      frame_tensor = maybe_frame.value();

    auto maybe_ts = msg.get<gxf::Timestamp>("timestamp");
    if (maybe_ts)
      input_timestamp = maybe_ts.value();
  }

  if (!step_tensor) {
    GXF_LOG_ERROR("No 'step' tensor found in input.");
    return GXF_FAILURE;
  }

  int32_t step = 0;
  if (CUDA_TRY(cudaMemcpyAsync(&step,
                               step_tensor->pointer(),
                               sizeof(int32_t),
                               cudaMemcpyDeviceToHost,
                               cuda_stream_handler_.getCudaStream()))) {
    return GXF_FAILURE;
  }
  if (CUDA_TRY(cudaStreamSynchronize(cuda_stream_handler_.getCudaStream()))) {
    return GXF_FAILURE;
  }

  // Select Context
  bool is_init = (step == 0);
  if (verbose_.get()) {
    GXF_LOG_INFO("Step: %d. Using %s Model.", step, is_init ? "Init" : "Fwd");
  }
  EngineContext& ctx = is_init ? init_engine_ctx_ : fwd_engine_ctx_;

  // Prepare Outputs
  auto result_msg = gxf::Entity::New(context());
  if (!result_msg)
    return result_msg.error();

  // BIND INPUTS
  for (size_t i = 0; i < ctx.input_tensor_names.size(); ++i) {
    const auto& name = ctx.input_tensor_names[i];
    const auto& binding_name = ctx.input_binding_names[i];
    const auto it = ctx.binding_infos.find(name);
    if (it == ctx.binding_infos.end()) {
      GXF_LOG_ERROR("No binding info found for input tensor '%s' (binding name: '%s')",
                    name.c_str(),
                    binding_name.c_str());
      return GXF_FAILURE;
    }
    const auto& info = it->second;

    void* ptr = nullptr;
    gxf::Shape shape;

    // Check if it is a state tensor
    bool is_state = false;
    for (const auto& s : state_tensor_names_.get()) {
      if (s == name) {
        is_state = true;
        break;
      }
    }

    if (is_state) {
      auto t = internal_states_.value().get<gxf::Tensor>(name.c_str());
      if (!t) {
        GXF_LOG_ERROR("Missing internal state %s", name.c_str());
        return GXF_FAILURE;
      }
      ptr = t.value()->pointer();
      shape = t.value()->shape();
    } else if (name == "query_points") {
      auto t = static_states_.value().get<gxf::Tensor>("query_points");
      if (!t) {
        GXF_LOG_ERROR("Missing query_points");
        return GXF_FAILURE;
      }
      ptr = t.value()->pointer();
      shape = t.value()->shape();
    } else {
      // Assume it comes from input message (e.g. frame)
      // Check "frame" or name provided
      gxf::Handle<gxf::Tensor> input_t;
      for (auto& m : messages) {
        // Try exact name first
        auto t = m.get<gxf::Tensor>(name.c_str());
        if (t) {
          input_t = t.value();
          break;
        }
      }
      if (!input_t && name == "video" && frame_tensor)
        input_t = frame_tensor;
      if (!input_t && name == "frame" && frame_tensor)
        input_t = frame_tensor;

      if (!input_t) {
        GXF_LOG_ERROR("Missing input tensor %s", name.c_str());
        return GXF_FAILURE;
      }
      ptr = input_t->pointer();
      shape = input_t->shape();
    }

    // Set Dimensions and Address
    nvinfer1::Dims dims;
    dims.nbDims = info.rank;

    int32_t rank_diff = static_cast<int32_t>(info.rank) - static_cast<int32_t>(shape.rank());
    if (rank_diff > 0) {
      if (verbose_.get()) {
        GXF_LOG_WARNING("Rank diff: %d for binding %s", rank_diff, binding_name.c_str());
      }
      for (int d = 0; d < rank_diff; ++d) dims.d[d] = 1;
      for (int d = 0; d < static_cast<int32_t>(shape.rank()); ++d)
        dims.d[d + rank_diff] = shape.dimension(d);
    } else {
      for (int d = 0; d < info.rank; ++d) dims.d[d] = shape.dimension(d);
    }

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
    ctx.cuda_execution_ctx->setBindingDimensions(info.index, dims);
    ctx.cuda_buffers[info.index] = ptr;
#else
    ctx.cuda_execution_ctx->setInputShape(binding_name.c_str(), dims);
    ctx.cuda_execution_ctx->setTensorAddress(binding_name.c_str(), ptr);
#endif
  }

  // BIND OUTPUTS
  for (size_t i = 0; i < ctx.output_tensor_names.size(); ++i) {
    const auto& name = ctx.output_tensor_names[i];
    const auto& binding_name = ctx.output_binding_names[i];
    const auto it = ctx.binding_infos.find(name);
    if (it == ctx.binding_infos.end()) {
      GXF_LOG_ERROR("Output tensor name '%s' not found in binding_infos", name.c_str());
      return GXF_FAILURE;
    }
    const auto& info = it->second;

    // Determine shape
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
    auto dims = ctx.cuda_execution_ctx->getBindingDimensions(info.index);
#else
    auto dims = ctx.cuda_execution_ctx->getTensorShape(binding_name.c_str());
#endif
    gxf::Shape shape{Dims2Dimensions(dims), info.rank};

    auto out_tensor = result_msg.value().add<gxf::Tensor>(name.c_str());
    if (!out_tensor)
      return out_tensor.error();

    auto res = out_tensor.value()->reshapeCustom(shape,
                                                 info.element_type,
                                                 gxf::PrimitiveTypeSize(info.element_type),
                                                 gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                                 gxf::MemoryStorageType::kDevice,
                                                 pool_);
    if (!res)
      return gxf::ToResultCode(res);

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
    ctx.cuda_buffers[info.index] = out_tensor.value()->pointer();
#else
    ctx.cuda_execution_ctx->setTensorAddress(binding_name.c_str(), out_tensor.value()->pointer());
#endif
  }

  // EXECUTE
#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
  if (!ctx.cuda_execution_ctx->enqueueV2(
          ctx.cuda_buffers.data(), cuda_stream_handler_.getCudaStream(), nullptr)) {
    GXF_LOG_ERROR("Failed to enqueue TensorRT inference");
    return GXF_FAILURE;
  }
#else
  if (!ctx.cuda_execution_ctx->enqueueV3(cuda_stream_handler_.getCudaStream())) {
    GXF_LOG_ERROR("Failed to enqueue TensorRT inference");
    return GXF_FAILURE;
  }
#endif

  // UPDATE STATE
  for (size_t i = 0; i < ctx.output_tensor_names.size(); ++i) {
    const auto& name = ctx.output_tensor_names[i];

    // Check if this output is a state
    bool is_state = false;
    for (const auto& s : state_tensor_names_.get()) {
      if (s == name) {
        is_state = true;
        break;
      }
    }

    if (verbose_.get()) {
      GXF_LOG_INFO("%s is state: %d", name.c_str(), is_state);
    }

    if (is_state) {
      // Copy from Output Tensor (in result_msg) to Internal State Tensor
      auto out_t_expected = result_msg.value().get<gxf::Tensor>(name.c_str());
      if (!out_t_expected) {
        GXF_LOG_ERROR("Output tensor %s not found", name.c_str());
        return GXF_FAILURE;
      }
      auto out_t = out_t_expected.value();

      auto maybe_state_t = internal_states_.value().get<gxf::Tensor>(name.c_str());
      gxf::Handle<gxf::Tensor> state_t;

      if (!maybe_state_t) {
        GXF_LOG_WARNING(
            "State tensor %s not found in internal states. Allocating new state tensor.",
            name.c_str());
        auto added_t = internal_states_.value().add<gxf::Tensor>(name.c_str());
        if (!added_t) {
          GXF_LOG_ERROR("Failed to add state tensor %s", name.c_str());
          return gxf::ToResultCode(added_t);
        }
        state_t = added_t.value();

        auto res = state_t->reshapeCustom(out_t->shape(),
                                          out_t->element_type(),
                                          out_t->bytes_per_element(),
                                          gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                          gxf::MemoryStorageType::kDevice,
                                          pool_);
        if (!res)
          return gxf::ToResultCode(res);
      } else {
        state_t = maybe_state_t.value();
        if (out_t->size() != state_t->size()) {
          GXF_LOG_ERROR("State size mismatch %s: Output %lu vs State %lu",
                        name.c_str(),
                        out_t->size(),
                        state_t->size());
          return GXF_FAILURE;
        }
      }

      if (CUDA_TRY(cudaMemcpyAsync(state_t->pointer(),
                                   out_t->pointer(),
                                   out_t->size(),
                                   cudaMemcpyDeviceToDevice,
                                   cuda_stream_handler_.getCudaStream()))) {
        return GXF_FAILURE;
      }
    }
  }

  // PUBLISH
  cuda_stream_handler_.toMessage(result_msg);
  // Pass through original step/window_num if they were in input
  if (step_tensor) {
    auto maybe_result_tensor = result_msg.value().add<nvidia::gxf::Tensor>("step");

    if (!maybe_result_tensor) {
      GXF_LOG_ERROR("Failed to allocate for output tensor step");
      return gxf::ToResultCode(maybe_result_tensor);
    }
    auto out_tensor = maybe_result_tensor.value();

    gxf::Shape shape{std::vector<int32_t>{1}};
    auto result = out_tensor->reshapeCustom(shape,
                                            step_tensor->element_type(),
                                            step_tensor->bytes_per_element(),
                                            gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                            gxf::MemoryStorageType::kDevice,
                                            pool_);

    if (!result) {
      GXF_LOG_ERROR("Failed to allocate for output tensor step");
      return gxf::ToResultCode(result);
    }

    if (CUDA_TRY(cudaMemcpyAsync(out_tensor->pointer(),
                                 step_tensor->pointer(),
                                 step_tensor->size(),
                                 cudaMemcpyDeviceToDevice,
                                 cuda_stream_handler_.getCudaStream()))) {
      return GXF_FAILURE;
    }
  }
  return gxf::ToResultCode(
      tx_->publish(result_msg.value(), input_timestamp ? input_timestamp->acqtime : 0));
}

gxf::Expected<std::vector<char>> TapNextInference::convertModelToEngine(
    const std::string& model_path, int32_t max_batch) {
  NvInferHandle<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(cuda_logger_));
  NvInferHandle<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
  config->setMaxWorkspaceSize(max_workspace_size_.get());
#else
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_.get());
#endif

  if (enable_fp16_.get()) {
#if NV_TENSORRT_MAJOR >= 10 && NV_TENSORRT_MINOR >= 13
    GXF_LOG_WARNING("FP16 mode is deprecated in TensorRT version %d.%d, ignoring.",
                    NV_TENSORRT_MAJOR,
                    NV_TENSORRT_MINOR);
#else
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif
  }

#if NV_TENSORRT_MAJOR < 10
  const auto explicitBatch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#else
  const auto explicitBatch = 1U;
#endif
  NvInferHandle<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
  NvInferHandle<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, cuda_logger_));

  if (!parser->parseFromFile(model_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    GXF_LOG_ERROR("Failed to parse ONNX: %s", model_path.c_str());
    return gxf::Unexpected{GXF_FAILURE};
  }

  // Optimization Profiles for dynamic shapes
  auto profile = builder->createOptimizationProfile();
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto input = network->getInput(i);
    auto dims = input->getDimensions();
    if (dims.d[0] == -1) {
      dims.d[0] = 1;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      dims.d[0] = max_batch;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
    }
  }
  config->addOptimizationProfile(profile);

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
  NvInferHandle<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
#else
  NvInferHandle<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
#endif

  if (!plan)
    return gxf::Unexpected{GXF_FAILURE};

  std::vector<char> result(plan->size());
  std::copy((const char*)plan->data(), (const char*)plan->data() + plan->size(), result.data());
  return result;
}

gxf::Expected<std::string> TapNextInference::queryHostEngineCapability(int dev_id) const {
  cudaDeviceProp device_prop = {0};
  if (cudaGetDeviceProperties(&device_prop, dev_id) != cudaSuccess)
    return gxf::Unexpected{};
  std::string name = replaceChar(device_prop.name, ' ', '-');
  std::stringstream ss;
  ss << name << "_c" << device_prop.major << device_prop.minor << "_n"
     << device_prop.multiProcessorCount;
  return ss.str();
}

gxf::Expected<std::string> TapNextInference::findEngineFilePath(
    const std::string& host_cap, const std::string& model_path) const {
  if (!IsValidDirectory(engine_cache_dir_.get())) {
    std::error_code ec;
    std::filesystem::create_directories(engine_cache_dir_.get(), ec);
    if (ec && !std::filesystem::is_directory(engine_cache_dir_.get())) {
      GXF_LOG_ERROR("Failed to create engine cache directory: %s", engine_cache_dir_.get().c_str());
      return gxf::Unexpected{};
    }
  }

  // Make unique hash based on model path to avoid collision between init and fwd if they have
  // different names
  std::filesystem::path p(model_path);
  std::string stem = p.stem().string();
  return engine_cache_dir_.get() + "/" + stem + "_" + host_cap + ".engine";
}

gxf_result_t TapNextInference::createQueryPoints(int32_t batch_size) {
  int32_t grid_size = grid_size_.get();
  int32_t height = grid_height_.get();
  int32_t width = grid_width_.get();

  // Create query points tensor
  auto qp = static_states_.value().add<gxf::Tensor>("query_points");
  if (!qp)
    return qp.error();

  int32_t num_points = grid_size * grid_size;
  gxf::Shape shape{std::vector<int32_t>{batch_size, num_points, 3}};

  if (grid_size < 2) {
    GXF_LOG_ERROR("Grid size must be at least 2, got %d", grid_size);
    return GXF_FAILURE;
  }

  // We allocate a temp host vector, write, then copy to Device.
  float margin = 8.0f;
  float h_step = (height - 2 * margin) / (grid_size - 1);
  float w_step = (width - 2 * margin) / (grid_size - 1);

  std::vector<float> host_data(batch_size * num_points * 3);
  for (int b = 0; b < batch_size; ++b) {
    int pt_idx = 0;
    for (int j = 0; j < grid_size;
         ++j) {  // x (width) - Outer loop to match Python meshgrid/flatten order
      for (int i = 0; i < grid_size; ++i) {  // y (height) - Inner loop
        int idx = b * num_points * 3 + pt_idx * 3;
        float y = margin + i * h_step;
        float x = margin + j * w_step;
        host_data[idx + 0] = 0.0f;
        host_data[idx + 1] = x;
        host_data[idx + 2] = y;
        pt_idx++;
      }
    }
  }

  // Allocate on Device
  auto res = qp.value()->reshapeCustom(shape,
                                       gxf::PrimitiveType::kFloat32,
                                       sizeof(float),
                                       gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                                       gxf::MemoryStorageType::kDevice,
                                       pool_);
  if (!res)
    return gxf::ToResultCode(res);

  if (CUDA_TRY(cudaMemcpy(qp.value()->pointer(),
                          host_data.data(),
                          host_data.size() * sizeof(float),
                          cudaMemcpyHostToDevice))) {
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

}  // namespace tapnext_inference
}  // namespace holoscan
}  // namespace nvidia
