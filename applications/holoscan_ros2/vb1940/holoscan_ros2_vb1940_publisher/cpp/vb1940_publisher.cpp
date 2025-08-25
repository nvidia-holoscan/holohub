/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/ros2/operators/publisher.hpp>

// ROS2 message includes
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>

// Hololink includes
#include <hololink/common/holoargs.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/tools.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>
#include <hololink/sensors/camera/vb1940/vb1940_mode.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel header
#include "convert_16bit_to_8bit_kernel.h"

class Vb1940PublisherOp : public holoscan::ros2::ops::PublisherOp<sensor_msgs::msg::Image> {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(Vb1940PublisherOp,
                                       holoscan::ros2::ops::PublisherOp<sensor_msgs::msg::Image>)

  Vb1940PublisherOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
    holoscan::ros2::ops::PublisherOp<sensor_msgs::msg::Image>::setup(spec);
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    // Receive entity data
    auto entity_result = op_input.receive<holoscan::gxf::Entity>("input");
    if (!entity_result) {
      HOLOSCAN_LOG_WARN("Failed to receive entity");
      return;
    }

    auto entity = entity_result.value();
    if (entity.is_null()) {
      HOLOSCAN_LOG_WARN("Received null entity");
      return;
    }

    // Use findAll to get all tensors in the entity
    const auto tensors = entity.nvidia::gxf::Entity::findAll<nvidia::gxf::Tensor>();
    if (tensors) {
      HOLOSCAN_LOG_DEBUG("Found {} tensors in entity", tensors.value().size());
      for (auto&& tensor : tensors.value()) {
        auto tensor_ptr = tensor.value();
        auto shape = tensor_ptr->shape();
        auto element_type = tensor_ptr->element_type();
        size_t bytes_size = tensor_ptr->bytes_size();

        // Initialize message metadata only once if not already done
        if (!message_initialized_) {
          message_.header.frame_id = "vb1940";
          message_.encoding = "rgb8";  // 8-bit RGB
          message_.is_bigendian = false;
          message_.height = shape.dimension(0);
          message_.width = shape.dimension(1);
          message_.step =
              shape.dimension(1) * 3 * sizeof(uint8_t);  // width * 3 channels * bytes_per_channel
          message_initialized_ = true;

          // Calculate expected bytes size for validation
          expected_bytes_size_ = bytes_size;

          // Calculate sizes for initial resize
          int input_channels = shape.dimension(2);  // RGB = 3 channels
          int output_channels = 3;                  // RGB = 3 channels
          size_t output_size =
              message_.width * message_.height * output_channels * sizeof(uint8_t);  // RGB8 size

          // Resize the message data buffer to match RGB8 size
          message_.data.resize(output_size);

          HOLOSCAN_LOG_INFO("Publishing frame: width: {}, height: {}, bytes_size: {}",
                            message_.width,
                            message_.height,
                            output_size);
        }

        // Assert that the incoming data size matches our expectations
        assert(bytes_size == expected_bytes_size_ && "Tensor size changed unexpectedly");

        // Calculate sizes
        int input_channels = shape.dimension(2);  // RGB = 3 channels
        int output_channels = 3;                  // RGB = 3 channels
        size_t input_size = bytes_size;           // Original RGB16 size
        size_t output_size =
            message_.width * message_.height * output_channels * sizeof(uint8_t);  // RGB8 size

        // Allocate CUDA buffer only once if not already allocated or if size changed
        if (!d_rgb8_buffer_ || output_size != rgb8_buffer_size_) {
          uint8_t* new_buffer;
          cudaMalloc(&new_buffer, output_size);
          d_rgb8_buffer_.reset(new_buffer);
          rgb8_buffer_size_ = output_size;
        }

        // Launch CUDA kernel to convert 16-bit to 8-bit per channel
        launch_convert_16bit_to_8bit_kernel(
            reinterpret_cast<const uint16_t*>(tensor_ptr->pointer()),
            d_rgb8_buffer_.get(),
            message_.width,
            message_.height,
            input_channels);

        message_.header.stamp = rclcpp::Clock().now();

        // Copy converted RGB8 data from device to host
        cudaMemcpy(message_.data.data(), d_rgb8_buffer_.get(), output_size, cudaMemcpyDeviceToHost);

        HOLOSCAN_LOG_TRACE(
            "Publishing Image: {}x{}x{} bytes", message_.height, message_.width, message_.step);
        publish(message_);
        HOLOSCAN_LOG_TRACE(
            "Published Image: {}x{}x{} bytes", message_.height, message_.width, message_.step);
      }
    } else {
      HOLOSCAN_LOG_DEBUG("  - No tensors found");
    }
  }

 private:
  std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> d_rgb8_buffer_{nullptr, [](uint8_t* ptr) {
                                                                           if (ptr)
                                                                             cudaFree(ptr);
                                                                         }};
  size_t rgb8_buffer_size_ = 0;       // RGB8 buffer size
  sensor_msgs::msg::Image message_;   // ROS2 message
  bool message_initialized_ = false;  // Flag to check if message is initialized
  size_t expected_bytes_size_ = 0;    // Expected bytes size for validation
};

class HoloscanVb1940PublisherApplication : public holoscan::Application {
 public:
  HoloscanVb1940PublisherApplication(CUcontext cuda_context, int cuda_device_ordinal,
                                     hololink::DataChannel& hololink_channel,
                                     const std::string& ibv_name, uint32_t ibv_port,
                                     std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera,
                                     hololink::sensors::vb1940_mode::Mode camera_mode,
                                     int frame_limit)
      : cuda_context_(cuda_context),
        cuda_device_ordinal_(cuda_device_ordinal),
        hololink_channel_(hololink_channel),
        ibv_name_(ibv_name),
        ibv_port_(ibv_port),
        camera_(camera),
        camera_mode_(camera_mode),
        frame_limit_(frame_limit) {}

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Condition> condition;
    if (frame_limit_) {
      condition = make_condition<CountCondition>("count", frame_limit_);
    } else {
      condition = make_condition<BooleanCondition>("ok", true);
    }
    camera_->set_mode(camera_mode_);

    auto csi_to_bayer_pool = make_resource<BlockMemoryPool>(
        "pool",
        1,  // storage_type of 1 is device memory
        camera_->get_width() * sizeof(uint16_t) * camera_->get_height(),  // block_size
        2);                                                               // num_blocks

    auto csi_to_bayer_operator = make_operator<hololink::operators::CsiToBayerOp>(
        "csi_to_bayer",
        Arg("allocator", csi_to_bayer_pool),
        Arg("cuda_device_ordinal", cuda_device_ordinal_));
    camera_->configure_converter(csi_to_bayer_operator);

    size_t frame_size = csi_to_bayer_operator->get_csi_length();
    auto receiver_operator = make_operator<hololink::operators::RoceReceiverOp>(
        "receiver",
        condition,
        Arg("frame_size", frame_size),
        Arg("frame_context", cuda_context_),
        Arg("ibv_name", ibv_name_),
        Arg("ibv_port", ibv_port_),
        Arg("hololink_channel", &hololink_channel_),
        Arg("device_start", std::function<void()>([this] { camera_->start(); })),
        Arg("device_stop", std::function<void()>([this] { camera_->stop(); })));

    auto pixel_format = static_cast<int>(camera_->get_pixel_format());
    auto bayer_format = static_cast<int>(camera_->get_bayer_format());
    auto image_processor_operator = make_operator<hololink::operators::ImageProcessorOp>(
        "image_processor",
        Arg("optical_black", 8),  // Optical black value for vb1940 is 8 for RAW10
        Arg("bayer_format", bayer_format),
        Arg("pixel_format", pixel_format));

    const int components_per_pixel = 3;
    auto bayer_pool =
        make_resource<BlockMemoryPool>("pool",
                                       1,  // storage_type of 1 is device memory
                                       camera_->get_width() * components_per_pixel *
                                           sizeof(uint16_t) * camera_->get_height(),  // block_size
                                       2);                                            // num_blocks

    auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
        "demosaic",
        Arg("pool", bayer_pool),
        Arg("generate_alpha", false),  // Output RGB instead of RGBA
        Arg("alpha_value", 65535),
        Arg("bayer_grid_pos", bayer_format),
        Arg("interpolation_mode", 0));

    auto ros2_bridge =
        make_resource<holoscan::ros2::Bridge>("vb1940_bridge_resource", "vb1940_bridge_node");
    auto vb1940_publisher =
        make_operator<Vb1940PublisherOp>("vb1940_publisher",
                                         condition,
                                         Arg("ros2_bridge", ros2_bridge),
                                         Arg("topic_name", std::string("vb1940/image")),
                                         Arg("qos", holoscan::ros2::QoS(10)));

    add_flow(receiver_operator, csi_to_bayer_operator, {{"output", "input"}});
    add_flow(csi_to_bayer_operator, image_processor_operator, {{"output", "input"}});
    add_flow(image_processor_operator, demosaic, {{"output", "receiver"}});
    add_flow(demosaic, vb1940_publisher, {{"transmitter", "input"}});
  }

 private:
  CUcontext cuda_context_;
  int cuda_device_ordinal_;
  hololink::DataChannel& hololink_channel_;
  std::string ibv_name_;
  uint32_t ibv_port_;
  std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera_;
  hololink::sensors::vb1940_mode::Mode camera_mode_;
  int frame_limit_;
};

int main(int argc, char** argv) {
  // Initialize ROS2
  rclcpp::init(argc, argv);

  using namespace hololink::args;
  OptionsDescription options_description("VB1940 Publisher Options");

  // Select the first available device to be the default
  std::string default_ibv_name;
  auto infiniband_devices = hololink::core::infiniband_devices();
  if (!infiniband_devices.empty())
    default_ibv_name = infiniband_devices[0];

  // clang-format off
    options_description.add_options()
        ("camera-mode",
         value<int>()->default_value(static_cast<int>(
             hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)),
         "VB1940 mode")
        ("frame-limit",
         value<int>()->default_value(0),
         "Exit after publishing this many frames")
        ("configuration",
         value<std::string>()->default_value("example_configuration.yaml"),
         "Configuration file")
        ("hololink",
         value<std::string>()->default_value("192.168.0.2"),
         "IP address of Hololink board")
        ("ibv-name",
         value<std::string>()->default_value(default_ibv_name),
         "IBV device to use")
        ("ibv-port",
         value<uint32_t>()->default_value(1),
         "Port number of IBV device (default: 1)");
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

    // Select the first available device if none specified
    std::string arg_ibv_name =
        variables_map["ibv-name"].empty() ? "" : variables_map["ibv-name"].as<std::string>();
    if (arg_ibv_name.empty()) {
      auto infiniband_devices = hololink::core::infiniband_devices();
      if (infiniband_devices.empty()) {
        HOLOSCAN_LOG_ERROR("No InfiniBand devices found");
        return -1;
      }
      arg_ibv_name = infiniband_devices[0];
    }

    // Get a handle to the Hololink device
    auto channel_metadata =
        hololink::Enumerator::find_channel(variables_map["hololink"].as<std::string>());
    hololink::DataChannel hololink_channel(channel_metadata);

    // Get a handle to the camera
    auto camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(hololink_channel);

    // Convert camera_mode to proper enum
    auto camera_mode =
        static_cast<hololink::sensors::vb1940_mode::Mode>(variables_map["camera-mode"].as<int>());

    // Set up the application
    auto app = std::make_unique<HoloscanVb1940PublisherApplication>(
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        variables_map["ibv-name"].as<std::string>(),
        variables_map["ibv-port"].as<uint32_t>(),
        camera,
        camera_mode,
        variables_map["frame-limit"].as<int>());
    app->config(variables_map["configuration"].as<std::string>());

    auto hololink = hololink_channel.hololink();
    hololink->start();
    hololink->reset();
    hololink->write_uint32(0x8, 0x0);  // Keep the sensor RESET at low
    camera->setup_clock();
    hololink->write_uint32(0x8, 0x1);  // Release the sensor RESET to high
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    camera->get_register_32(0x0000);  // DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera->get_register_32(0x0734);  // EXT_CLOCK(25MHz = 0x017d7840)
    camera->configure(camera_mode);

    app->run();
    hololink->stop();

    cuDevicePrimaryCtxRelease(cu_device);
    // Shutdown ROS2
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
    // Ensure ROS2 is shutdown even if an error occurs
    if (rclcpp::ok())
      rclcpp::shutdown();
    return -1;
  }
}
