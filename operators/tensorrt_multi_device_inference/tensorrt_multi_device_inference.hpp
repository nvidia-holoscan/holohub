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
#ifndef HOLOHUB_OPERATORS_TENSORRT_MULTI_DEVICE_INFERENCE_HPP
#define HOLOHUB_OPERATORS_TENSORRT_MULTI_DEVICE_INFERENCE_HPP

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "holoscan/holoscan.hpp"

#include "multidevice.hpp"

namespace holoscan::ops {

/**
 * @brief Runs a single TensorRT engine sharded across >=2 GPUs (TensorRT
 * Multi-Device, TRT-28040), so one operator drives N GPUs via NCCL.
 *
 * Self-contained: wraps the validated `holoscan::inference::MultiDeviceTrt`
 * runtime (ncclCommInitAll -> per-rank deserialize -> concurrent setCommunicator
 * -> host-bounce input replication -> fan-out enqueueV3 -> rank-0 output). Does
 * not depend on the SDK's HoloInfer/InferenceOp.
 *
 * Engine layout (set via `engine_paths`):
 *  - one path  -> a single offline-sharded plan deserialized on every rank;
 *  - N paths   -> per-rank weight-shard plans (tensor/Megatron parallelism),
 *                 index == rank.
 *
 * Requires TensorRT >= 11.0 (Multi-Device GA), NCCL, and >= 2 homogeneous GPUs
 * (SM80+). Input/output are single FP32 device tensors carried on a GXF entity.
 */
class TensorRtMultiDeviceInferenceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorRtMultiDeviceInferenceOp)

  TensorRtMultiDeviceInferenceOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;

  Parameter<std::vector<std::string>> engine_paths_;  // index == rank
  Parameter<std::vector<int32_t>> device_ids_;        // physical GPU per rank
  Parameter<std::string> input_tensor_name_;
  Parameter<std::string> output_tensor_name_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  // TensorRT rank-0 resources (ranks 1..N-1 live inside md_).
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<holoscan::inference::MultiDeviceTrt> md_;

  std::string trt_input_name_;
  std::string trt_output_name_;
  std::vector<int64_t> output_shape_;
  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  void* d_input_ = nullptr;
  void* d_output_ = nullptr;
};

}  // namespace holoscan::ops

#endif  // HOLOHUB_OPERATORS_TENSORRT_MULTI_DEVICE_INFERENCE_HPP
