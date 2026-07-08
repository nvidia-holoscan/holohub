/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <holoscan/core/parameter.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/ros2/operators/subscriber.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

// ROS2 message includes
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <hololink/common/holoargs.hpp>

/**
 * @brief Advanced Holoscan operator that subscribes to IMX274 camera images from ROS2.
 *
 * This operator demonstrates production-ready integration between ROS2 and Holoscan for
 * high-performance camera visualization and processing. It subscribes to ROS2
 * sensor_msgs::Image messages (typically from IMX274 camera data), converts them back to
 * Holoscan tensor format, and outputs them for visualization or further processing in the
 * Holoscan pipeline.
 *
 * Key features:
 * - Efficient ROS2 message to Holoscan tensor conversion
 * - GPU memory management with device allocation
 * - Host-to-device memory transfers for zero-copy processing
 * - Integration with Holoviz for real-time visualization
 *
 * The operator is designed to complement Imx274PublisherOp, enabling distributed camera
 * processing workflows where image capture, processing, and visualization can occur
 * across different nodes in a robotics system.
 */
class Imx274SubscriberOp : public holoscan::ros2::ops::SubscriberOp<sensor_msgs::msg::Image> {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(Imx274SubscriberOp,
                                       holoscan::ros2::ops::SubscriberOp<sensor_msgs::msg::Image>)

  Imx274SubscriberOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.param(pool_, "pool", "Pool", "Pool to allocate the output message.");
    spec.output<holoscan::gxf::Entity>("output");
    holoscan::ros2::ops::SubscriberOp<sensor_msgs::msg::Image>::setup(spec);
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    // Receive ROS2 message - this is a blocking call
    auto message = receive().get();

    // Create allocator handle
    auto allocator_handle =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), pool_->gxf_cid());
    if (!allocator_handle) {
      HOLOSCAN_LOG_ERROR("Failed to create allocator handle");
      return;
    }
    // Create output message with tensor
    auto out_message =
        nvidia::gxf::CreateTensorMap(context.context(),
                                     allocator_handle.value(),
                                     {{"",
                                       nvidia::gxf::MemoryStorageType::kDevice,
                                       nvidia::gxf::Shape{static_cast<int32_t>(message.height),
                                                          static_cast<int32_t>(message.width),
                                                          static_cast<int32_t>(3)},  // RGB channels
                                       nvidia::gxf::PrimitiveType::kUnsigned8,  // 8-bit per channel
                                       0,
                                       nvidia::gxf::ComputeTrivialStrides(
                                           nvidia::gxf::Shape{static_cast<int32_t>(message.height),
                                                              static_cast<int32_t>(message.width),
                                                              static_cast<int32_t>(3)},
                                           sizeof(uint8_t))}},
                                     false);

    if (!out_message) {
      HOLOSCAN_LOG_ERROR("Failed to create output tensor map. Error code: {}",
                         static_cast<int>(out_message.error()));
      return;
    }

    // Get the tensor and copy data
    auto maybe_tensor = out_message.value().get<nvidia::gxf::Tensor>("");
    if (!maybe_tensor) {
      HOLOSCAN_LOG_ERROR("Failed to get output tensor. Error code: {}",
                         static_cast<int>(maybe_tensor.error()));
      return;
    }

    // Copy data to the tensor
    cudaMemcpy(maybe_tensor.value()->pointer(),
               message.data.data(),
               message.data.size(),
               cudaMemcpyHostToDevice);

    // Create a new Holoscan entity from the GXF entity and emit it
    auto result = holoscan::gxf::Entity(std::move(out_message.value()));
    op_output.emit(result, "output");
  }

 private:
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> pool_;
};

class HoloscanImx274SubscriberApplication : public holoscan::Application {
 public:
  HoloscanImx274SubscriberApplication(bool headless, bool fullscreen)
      : headless_(headless), fullscreen_(fullscreen) {}

  void compose() override {
    using namespace holoscan;

    auto ros2_bridge = make_resource<holoscan::ros2::Bridge>("imx274_subscriber_bridge_resource",
                                                             "imx274_subscriber_bridge_node");
    auto tensor_pool = make_resource<BlockMemoryPool>(
        "pool",
        1,                                  // storage_type of 1 is device memory
        3840 * 2160 * 3 * sizeof(uint8_t),  // block_size = Max size for RGB8 (4K mode)
        2);                                 // num_blocks
    auto subscriber = make_operator<Imx274SubscriberOp>(
        "imx274_subscriber",
        holoscan::Arg("ros2_bridge", ros2_bridge),
        holoscan::Arg("topic_name", std::string("imx274/image")),
        holoscan::Arg("qos", holoscan::ros2::QoS(10)),
        holoscan::Arg("pool", tensor_pool),
        holoscan::Arg("message_queue_max_size",
                      Imx274SubscriberOp::Subscriber::MessageQueue::size_type(3)));

    auto visualizer = make_operator<holoscan::ops::HolovizOp>("holoviz",
                                                              Arg("fullscreen", fullscreen_),
                                                              Arg("headless", headless_),
                                                              Arg("framebuffer_srgb", true),
                                                              Arg("enable_cuda_interop", true));

    add_flow(subscriber, visualizer, {{"output", "receivers"}});
  }

 private:
  bool headless_;
  bool fullscreen_;
};

int main(int argc, char** argv) {
  // Initialize ROS2
  rclcpp::init(argc, argv);

  using namespace hololink::args;
  OptionsDescription options_description("IMX274 Subscriber Options");

  // clang-format off
    options_description.add_options()
        ("headless", value<bool>()->default_value(false), "Run in headless mode")
        ("fullscreen", value<bool>()->default_value(false), "Run in fullscreen mode")
        ("configuration",
         value<std::string>()->default_value(""),
         "Configuration file (optional)");
  // clang-format on

  auto variables_map = Parser().parse_command_line(argc, argv, options_description);

  try {
    // Initialize CUDA
    cuInit(0);
    CUdevice cu_device;
    int cu_device_ordinal = 0;
    cuDeviceGet(&cu_device, cu_device_ordinal);
    CUcontext cu_context;
    cuDevicePrimaryCtxRetain(&cu_context, cu_device);

    // Set up the application
    auto app = std::make_unique<HoloscanImx274SubscriberApplication>(
        variables_map["headless"].as<bool>(), variables_map["fullscreen"].as<bool>());
    const auto configuration = variables_map["configuration"].as<std::string>();
    if (!configuration.empty()) {
      app->config(configuration);
    }

    app->run();

    cuDevicePrimaryCtxRelease(cu_device);
    // Shutdown ROS2
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
    rclcpp::shutdown();
    return -1;
  }
}
