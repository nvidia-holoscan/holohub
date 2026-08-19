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

// Minimal demonstration of the TensorRtMultiDeviceInferenceOp: a deterministic
// tensor source -> the multi-GPU (TensorRT Multi-Device) inference operator ->
// a sink that copies the result to host and prints a checksum. Requires a
// TensorRT-11 + NCCL container and >= 2 GPUs; engine plan(s) sharded offline.

#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gxf/std/allocator.hpp>
#include <gxf/std/tensor.hpp>

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/holoscan.hpp"

#include "tensorrt_multi_device_inference.hpp"

namespace holoscan::ops {

// Emits a single deterministic FP32 device tensor of the configured shape.
class DeterministicTensorSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DeterministicTensorSourceOp)
  DeterministicTensorSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("out");
    spec.param(rows_, "rows", "Rows", "Input tensor rows", 2048);
    spec.param(cols_, "cols", "Cols", "Input tensor cols", 4096);
    spec.param(
        tensor_name_, "tensor_name", "Tensor name", "Output tensor name", std::string("input"));
    spec.param(allocator_, "allocator", "Allocator", "Allocator");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    const int64_t n = static_cast<int64_t>(rows_.get()) * cols_.get();
    std::vector<float> host(n);
    for (int64_t i = 0; i < n; ++i) { host[i] = 0.02f * static_cast<float>((i % 31) - 15); }

    auto entity = nvidia::gxf::Entity::New(context.context());
    auto tensor = entity.value().add<nvidia::gxf::Tensor>(tensor_name_.get().c_str());
    auto alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                     allocator_->gxf_cid());
    tensor.value()->reshape<float>(nvidia::gxf::Shape({rows_.get(), cols_.get()}),
                                   nvidia::gxf::MemoryStorageType::kDevice,
                                   alloc.value());
    cudaMemcpy(tensor.value()->pointer(), host.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    auto msg = holoscan::gxf::Entity(std::move(entity.value()));
    op_output.emit(msg, "out");
  }

 private:
  Parameter<int> rows_;
  Parameter<int> cols_;
  Parameter<std::string> tensor_name_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

// Copies the result to host and prints a checksum so the run is verifiable.
class ChecksumSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ChecksumSinkOp)
  ChecksumSinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("in");
    spec.param(
        tensor_name_, "tensor_name", "Tensor name", "Input tensor name", std::string("output"));
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto msg = op_input.receive<holoscan::gxf::Entity>("in").value();
    auto tensor = msg.get<holoscan::Tensor>(tensor_name_.get().c_str());
    if (!tensor) {
      throw std::runtime_error("ChecksumSinkOp: output tensor not found");
    }
    const size_t n = tensor->size();
    std::vector<float> host(n);
    cudaMemcpy(host.data(), tensor->data(), n * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = std::accumulate(host.begin(), host.end(), 0.0);
    HOLOSCAN_LOG_INFO("Multi-Device inference output: {} elements, checksum {:.6f}", n, sum);
    printf("Multi-Device inference complete: %zu elements, checksum %.6f\n", n, sum);
  }

 private:
  Parameter<std::string> tensor_name_;
};

}  // namespace holoscan::ops

class MultiDeviceInferenceApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    const auto engine_paths = from_config("inference.engine_paths").as<std::vector<std::string>>();
    const auto device_ids = from_config("inference.device_ids").as<std::vector<int32_t>>();

    auto pool = make_resource<UnboundedAllocator>("pool");
    auto source =
        make_operator<ops::DeterministicTensorSourceOp>("source",
                                                        from_config("source"),
                                                        Arg("allocator") = pool,
                                                        make_condition<CountCondition>(1));
    auto inference = make_operator<ops::TensorRtMultiDeviceInferenceOp>(
        "inference",
        Arg("engine_paths") = engine_paths,
        Arg("device_ids") = device_ids,
        Arg("input_tensor_name") = std::string("input"),
        Arg("output_tensor_name") = std::string("output"),
        Arg("allocator") = pool);
    auto sink = make_operator<ops::ChecksumSinkOp>("sink");

    add_flow(source, inference);
    add_flow(inference, sink);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MultiDeviceInferenceApp>();
  std::string config = "multi_device_inference.yaml";
  if (argc > 1) {
    config = argv[1];
  }
  app->config(config);
  app->run();
  return 0;
}
