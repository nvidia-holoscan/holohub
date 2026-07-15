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
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

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
    // Wait for the next ROS2 message with a timeout instead of blocking forever:
    // Holoviz only services its window events when it receives a frame, so an
    // indefinitely blocked receive freezes the window ("not responding") until
    // the publisher starts streaming. Keep the pending future across compute()
    // calls: abandoning a timed-out future would leave its promise queued in the
    // bridge, silently consuming the next message.
    if (!pending_message_) {
      pending_message_ = receive();
    }
    if (pending_message_->wait_for(kReceiveTimeout) == std::future_status::ready) {
      auto message = pending_message_->get();
      pending_message_.reset();

      // Validate the image layout before allocating and copying
      const size_t expected_step = static_cast<size_t>(message.width) * 3 * sizeof(uint8_t);
      const size_t expected_size = expected_step * message.height;
      if (message.encoding != "rgb8" || message.step != expected_step ||
          message.data.size() != expected_size) {
        HOLOSCAN_LOG_ERROR(
            "Skipping image with unexpected layout: encoding='{}', step={}, data_size={} "
            "(expected rgb8, step={}, data_size={})",
            message.encoding,
            message.step,
            message.data.size(),
            expected_step,
            expected_size);
      } else {
        last_message_ = std::move(message);
      }
    }

    if (last_message_) {
      // Emit the most recent frame (re-emitted while no new frame has arrived
      // so the Holoviz window keeps servicing events).
      emit_frame(last_message_->width, last_message_->height, last_message_->data.data(),
                 op_output, context);
    } else {
      // No frame received yet: emit a black placeholder so the Holoviz window
      // stays responsive while waiting for the publisher to start.
      emit_frame(kPlaceholderWidth, kPlaceholderHeight, nullptr, op_output, context);
    }
  }

 private:
  void emit_frame(uint32_t width, uint32_t height, const uint8_t* host_data,
                  holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) {
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
                                       nvidia::gxf::Shape{static_cast<int32_t>(height),
                                                          static_cast<int32_t>(width),
                                                          static_cast<int32_t>(3)},  // RGB channels
                                       nvidia::gxf::PrimitiveType::kUnsigned8,  // 8-bit per channel
                                       0,
                                       nvidia::gxf::ComputeTrivialStrides(
                                           nvidia::gxf::Shape{static_cast<int32_t>(height),
                                                              static_cast<int32_t>(width),
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

    // Copy data to the tensor (or clear it to black when no frame is available)
    const size_t frame_bytes = static_cast<size_t>(width) * height * 3 * sizeof(uint8_t);
    cudaError_t copy_status;
    if (host_data != nullptr) {
      copy_status = cudaMemcpy(
          maybe_tensor.value()->pointer(), host_data, frame_bytes, cudaMemcpyHostToDevice);
    } else {
      copy_status = cudaMemset(maybe_tensor.value()->pointer(), 0, frame_bytes);
    }
    if (copy_status != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to fill output tensor: {}; skipping frame",
                         cudaGetErrorString(copy_status));
      return;
    }

    // Create a new Holoscan entity from the GXF entity and emit it
    auto result = holoscan::gxf::Entity(std::move(out_message.value()));
    op_output.emit(result, "output");
  }

  static constexpr std::chrono::milliseconds kReceiveTimeout{100};
  static constexpr uint32_t kPlaceholderWidth = 3840;
  static constexpr uint32_t kPlaceholderHeight = 2160;

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> pool_;
  std::optional<std::future<MessageType>> pending_message_;
  std::optional<MessageType> last_message_;
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
    auto check_cuda = [](CUresult result, const char* what) {
      if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed (" +
                                 std::to_string(static_cast<int>(result)) + ")");
      }
    };
    // Initialize CUDA
    check_cuda(cuInit(0), "cuInit");
    CUdevice cu_device;
    int cu_device_ordinal = 0;
    check_cuda(cuDeviceGet(&cu_device, cu_device_ordinal), "cuDeviceGet");
    CUcontext cu_context;
    check_cuda(cuDevicePrimaryCtxRetain(&cu_context, cu_device), "cuDevicePrimaryCtxRetain");
    // Release the primary context on all exit paths, including exceptions
    struct CudaContextGuard {
      CUdevice device;
      ~CudaContextGuard() { cuDevicePrimaryCtxRelease(device); }
    } cuda_context_guard{cu_device};

    // Set up the application
    auto app = std::make_unique<HoloscanImx274SubscriberApplication>(
        variables_map["headless"].as<bool>(), variables_map["fullscreen"].as<bool>());
    const auto configuration = variables_map["configuration"].as<std::string>();
    if (!configuration.empty()) {
      app->config(configuration);
    }

    app->run();

    // Shutdown ROS2 (CUDA context release runs via the scope guard)
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
    rclcpp::shutdown();
    return -1;
  }
}
