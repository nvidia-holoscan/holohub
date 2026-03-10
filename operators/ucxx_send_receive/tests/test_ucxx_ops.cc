/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <random>

#include <holoscan/holoscan.hpp>

#include <operators/ucxx_send_receive/ucxx_endpoint.hpp>
#include <operators/ucxx_send_receive/receiver_op/ucxx_receiver_op.hpp>
#include <operators/ucxx_send_receive/sender_op/ucxx_sender_op.hpp>

constexpr int kImageWidth = 1920;
constexpr int kImageHeight = 1080;
constexpr int kImageChannels = 3;

// Emits a Holoscan Tensor with random data.
class TensorTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorTxOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.param(allocator_, "allocator", "Allocator", "Memory allocator for tensor");
    spec.output<holoscan::gxf::Entity>("out");
  }

  void compute([[maybe_unused]] holoscan::InputContext& input, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override {
    // Create a GXF entity
    auto out_message = holoscan::gxf::Entity::New(&context);

    // Add a tensor to the entity (need to cast to nvidia::gxf::Entity&)
    auto tensor = static_cast<nvidia::gxf::Entity&>(out_message)
                      .add<nvidia::gxf::Tensor>("").value();

    // Get the allocator handle
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());

    // Reshape tensor with dimensions [height, width, channels]
    tensor->reshape<uint8_t>(
        nvidia::gxf::Shape{kImageHeight, kImageWidth, kImageChannels},
        nvidia::gxf::MemoryStorageType::kDevice,
        gxf_allocator.value());

    // Fill with random data (for testing purposes, use a simple pattern)
    const size_t tensor_size = kImageHeight * kImageWidth * kImageChannels;
    std::vector<uint8_t> host_data(tensor_size);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < tensor_size; ++i) {
      host_data[i] = static_cast<uint8_t>(dis(gen));
    }

    // Copy data to device
    cudaError_t err = cudaMemcpy(tensor->pointer(),
      host_data.data(),
      tensor_size,
      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("cudaMemcpy failed with error: {}", cudaGetErrorString(err));
      throw std::runtime_error("cudaMemcpy failed in TensorTxOp");
    }

    output.emit(out_message, "out");
  }

 private:
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

// Receives and verifies Tensor.
class TensorRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorRxOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("in");
  }

  void compute(holoscan::InputContext& input, [[maybe_unused]] holoscan::OutputContext& output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto entity = input.receive<holoscan::gxf::Entity>("in");
    if (!entity.has_value()) {
      HOLOSCAN_LOG_ERROR("Failed to receive entity");
      return;
    }

    // Get the tensor from the entity
    auto tensor = entity.value().get<holoscan::Tensor>("");
    if (!tensor) {
      HOLOSCAN_LOG_ERROR("Failed to get tensor from entity");
      return;
    }

    // Verify tensor dimensions
    HOLOSCAN_LOG_INFO("Received tensor with rank {} and shape: [{}, {}, {}]",
                      tensor->ndim(), tensor->shape()[0], tensor->shape()[1], tensor->shape()[2]);

    // Verify data type
    HOLOSCAN_LOG_INFO("Received tensor with dtype: {} bits: {}",
                      tensor->dtype().code, tensor->dtype().bits);
  }
};

class UcxxTestApp : public holoscan::Application {
 public:
  UcxxTestApp() {}

  void compose() override {
    // Create allocators
    auto tx_allocator = make_resource<holoscan::UnboundedAllocator>("tx_allocator");
    auto rx_allocator = make_resource<holoscan::UnboundedAllocator>("rx_allocator");

    // Create UCXX endpoints
    auto ucxx_server_endpoint = make_resource<holoscan::ops::UcxxEndpoint>(
        "ucxx_server_endpoint", holoscan::Arg("port", 50009), holoscan::Arg("listen", true));

    auto ucxx_client_endpoint = make_resource<holoscan::ops::UcxxEndpoint>(
        "ucxx_client_endpoint", holoscan::Arg("port", 50009), holoscan::Arg("listen", false));

    // Create operators
    auto tensor_tx = make_operator<TensorTxOp>(
        "tensor_tx",
        holoscan::Arg("allocator", tx_allocator),
        make_condition<holoscan::CountCondition>(10));

    auto ucxx_tx = make_operator<holoscan::ops::UcxxSenderOp>(
        "ucxx_tx",
        holoscan::Arg("tag", 777ul),
        holoscan::Arg("endpoint", ucxx_client_endpoint));

    auto ucxx_rx = make_operator<holoscan::ops::UcxxReceiverOp>(
        "ucxx_rx",
        holoscan::Arg("tag", 777ul),
        holoscan::Arg("buffer_size", (4 << 10) + kImageWidth * kImageHeight * kImageChannels),
        holoscan::Arg("endpoint", ucxx_server_endpoint),
        holoscan::Arg("allocator", rx_allocator),
        make_condition<holoscan::CountCondition>(11));  // 0th call initiates receive request

    auto tensor_rx = make_operator<TensorRxOp>(
        "tensor_rx",
        make_condition<holoscan::CountCondition>(10));

    // Connect operators
    add_flow(tensor_tx, ucxx_tx);
    add_flow(ucxx_rx, tensor_rx);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<UcxxTestApp>();

  if (argc >= 2) {
    app->config(argv[1]);
  }

  app->run();
  return 0;
}
