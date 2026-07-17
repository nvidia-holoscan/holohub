/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_multi_device_inference.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <utility>

#include <gxf/std/allocator.hpp>
#include <gxf/std/tensor.hpp>

#include "holoscan/core/gxf/entity.hpp"

namespace holoscan::ops {

namespace {
// TensorRT logger (errors only).
class TrtLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kERROR) {
      HOLOSCAN_LOG_ERROR("[TensorRT] {}", msg);
    }
  }
};
TrtLogger g_trt_logger;

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: cannot open engine: " + path);
  }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf(n);
  f.read(buf.data(), n);
  return buf;
}

int64_t volume(const nvinfer1::Dims& d) {
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) { v *= d.d[i]; }
  return v;
}
}  // namespace

void TensorRtMultiDeviceInferenceOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("in").connector(IOSpec::ConnectorType::kDefault);
  spec.output<holoscan::gxf::Entity>("out");

  spec.param(in_, "in", "Input", "Input port", &spec.inputs()["in"]);
  spec.param(out_, "out", "Output", "Output port", &spec.outputs()["out"]);
  spec.param(engine_paths_,
             "engine_paths",
             "Engine paths",
             "TensorRT engine plan files, one per rank (index == rank). A single path is a shared "
             "offline-sharded plan; N paths are per-rank weight-shard plans.");
  spec.param(device_ids_,
             "device_ids",
             "Device ids",
             "Physical GPU id per rank; device_ids[0] is rank 0.");
  spec.param(input_tensor_name_,
             "input_tensor_name",
             "Input tensor name",
             "Name of the input tensor on the incoming message.",
             std::string("input"));
  spec.param(output_tensor_name_,
             "output_tensor_name",
             "Output tensor name",
             "Name of the output tensor on the emitted message.",
             std::string("output"));
  spec.param(allocator_, "allocator", "Allocator", "Output tensor allocator.");
}

void TensorRtMultiDeviceInferenceOp::initialize() {
  Operator::initialize();
}

void TensorRtMultiDeviceInferenceOp::start() {
  const auto& paths = engine_paths_.get();
  const auto& devices = device_ids_.get();
  if (devices.size() < 2) {
    throw std::runtime_error(
        "TensorRtMultiDeviceInferenceOp requires >= 2 device_ids (Multi-Device is multi-GPU).");
  }
  if (paths.empty()) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp requires at least one engine path.");
  }

  // Rank 0 lives in this operator; deserialize its plan on device_ids[0].
  if (cudaSetDevice(devices[0]) != cudaSuccess) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: cudaSetDevice(rank0) failed.");
  }
  std::vector<char> rank0_bytes = read_file(paths.front());
  runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
  engine_.reset(runtime_->deserializeCudaEngine(rank0_bytes.data(), rank0_bytes.size()));
  if (!engine_) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: rank0 deserialize failed.");
  }
  context_.reset(engine_->createExecutionContext());
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: cudaStreamCreate failed.");
  }

  // Discover the single input/output tensor (FP32) from the rank-0 engine.
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* nm = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT) {
      trt_input_name_ = nm;
    } else {
      trt_output_name_ = nm;
    }
  }
  const nvinfer1::Dims in_dims = engine_->getTensorShape(trt_input_name_.c_str());
  const nvinfer1::Dims out_dims = engine_->getTensorShape(trt_output_name_.c_str());
  input_bytes_ = static_cast<size_t>(volume(in_dims)) * sizeof(float);
  output_bytes_ = static_cast<size_t>(volume(out_dims)) * sizeof(float);
  output_shape_.assign(out_dims.d, out_dims.d + out_dims.nbDims);

  if (cudaMalloc(&d_output_, output_bytes_) != cudaSuccess) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: cudaMalloc(output) failed.");
  }

  // Bring up the other ranks (1..N-1) and attach the NCCL communicator to all ranks.
  const std::vector<int> dev_ids(devices.begin(), devices.end());
  md_ = std::make_unique<holoscan::inference::MultiDeviceTrt>();
  bool ok = false;
  if (paths.size() == 1) {
    ok = md_->initialize(rank0_bytes, dev_ids, context_.get(), g_trt_logger);
  } else {
    std::vector<std::vector<char>> per_rank;
    per_rank.reserve(paths.size());
    for (const auto& p : paths) { per_rank.push_back(read_file(p)); }
    ok = md_->initialize_per_rank(per_rank, dev_ids, context_.get(), g_trt_logger);
  }
  if (!ok) {
    throw std::runtime_error(
        "TensorRtMultiDeviceInferenceOp: MultiDeviceTrt initialization failed.");
  }
  HOLOSCAN_LOG_INFO("TensorRtMultiDeviceInferenceOp ready: {} ranks, input '{}', output '{}'",
                    md_->world_size(),
                    trt_input_name_,
                    trt_output_name_);
}

void TensorRtMultiDeviceInferenceOp::compute(InputContext& op_input, OutputContext& op_output,
                                             ExecutionContext& context) {
  auto in_message = op_input.receive<holoscan::gxf::Entity>("in").value();
  auto in_tensor = in_message.get<holoscan::Tensor>(input_tensor_name_.get().c_str());
  if (!in_tensor) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: input tensor '" +
                             input_tensor_name_.get() + "' not found.");
  }

  cudaSetDevice(device_ids_.get()[0]);
  void* in_ptr = in_tensor->data();

  // Rank 0 reads the incoming tensor directly; the MD runtime replicates it to the
  // other ranks (host-bounce), runs enqueueV3 on every rank, and rendezvous via NCCL.
  context_->setTensorAddress(trt_input_name_.c_str(), in_ptr);
  context_->setTensorAddress(trt_output_name_.c_str(), d_output_);

  std::map<std::string, std::pair<const void*, size_t>> md_inputs{
      {trt_input_name_, {in_ptr, input_bytes_}}};
  md_->enqueue_others(context_.get(), md_inputs);
  const bool rank0_ok = context_->enqueueV3(stream_);
  const bool others_ok = md_->wait();
  cudaStreamSynchronize(stream_);
  if (!rank0_ok || !others_ok) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: multi-device enqueue failed.");
  }

  // Emit the rank-0 output as a new device tensor.
  auto out_entity = nvidia::gxf::Entity::New(context.context());
  if (!out_entity) {
    throw std::runtime_error("TensorRtMultiDeviceInferenceOp: output Entity::New failed.");
  }
  auto gxf_tensor = out_entity.value().add<nvidia::gxf::Tensor>(output_tensor_name_.get().c_str());
  auto gxf_alloc =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  std::vector<int32_t> shape32(output_shape_.begin(), output_shape_.end());
  gxf_tensor.value()->reshape<float>(
      nvidia::gxf::Shape(shape32), nvidia::gxf::MemoryStorageType::kDevice, gxf_alloc.value());
  cudaMemcpyAsync(
      gxf_tensor.value()->pointer(), d_output_, output_bytes_, cudaMemcpyDeviceToDevice, stream_);
  cudaStreamSynchronize(stream_);

  auto out_message = holoscan::gxf::Entity(std::move(out_entity.value()));
  op_output.emit(out_message, "out");
}

void TensorRtMultiDeviceInferenceOp::stop() {
  md_.reset();  // tears down ranks 1..N-1 first
  if (d_output_) {
    cudaFree(d_output_);
    d_output_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

}  // namespace holoscan::ops
