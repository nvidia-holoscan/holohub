/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fstream>
#include <memory>
#include <vector>
#include <filesystem>

#include "holoscan/holoscan.hpp"
#include "gxf/core/gxf.h"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "advanced_network/common.h"
#include "adv_network_media_rx.h"

using namespace holoscan::advanced_network;

#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cudaError_t cuda_status = stmt;                                                     \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         static_cast<int>(cuda_status));                                \
      throw std::runtime_error("CUDA operation failed");                                \
    }                                                                                   \
  }

namespace holoscan::ops {

/**
 * @class FramesWriterOp
 * @brief Operator for writing frame data to file.
 *
 * This operator can handle:
 * - Input types: VideoBuffer, GXF Tensor
 * - Memory sources: Host memory (kHost, kSystem) and Device memory (kDevice)
 * - Automatic memory type detection and appropriate copying (device-to-host when needed)
 *
 * Features:
 * - Automatic input type detection (VideoBuffer takes precedence, then GXF Tensor)
 * - Efficient memory handling with reusable host buffer for device-to-host copies
 * - Comprehensive error handling and logging
 * - Binary file output with proper stream management
 *
 * Parameters:
 * - num_of_frames_to_record: Number of frames to write before stopping
 * - file_path: Output file path (default: "./output.bin")
 */
class FramesWriterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FramesWriterOp)

  FramesWriterOp() = default;

  ~FramesWriterOp() {
    if (file_stream_.is_open()) { file_stream_.close(); }
  }

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
    spec.param(num_of_frames_to_record_,
               "num_of_frames_to_record",
               "The number of frames to write to file");
    spec.param<std::string>(
        file_path_, "file_path", "Output File Path", "Path to the output file", "./output.bin");
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("FramesWriterOp::initialize()");
    holoscan::Operator::initialize();

    std::string file_path = file_path_.get();
    HOLOSCAN_LOG_INFO("Original file path from config: {}", file_path);
    HOLOSCAN_LOG_INFO("Current working directory: {}", std::filesystem::current_path().string());

    // Convert to absolute path if relative
    std::filesystem::path path(file_path);
    if (!path.is_absolute()) {
      path = std::filesystem::absolute(path);
      file_path = path.string();
      HOLOSCAN_LOG_INFO("Converted to absolute path: {}", file_path);
    }

    HOLOSCAN_LOG_INFO("Attempting to open output file: {}", file_path);
    file_stream_.open(file_path, std::ios::out | std::ios::binary);
    if (!file_stream_) {
      HOLOSCAN_LOG_ERROR("Failed to open output file: {}. Check permissions and disk space.",
                         file_path);

      // Additional debugging information
      std::error_code ec;
      if (path.has_parent_path()) {
        auto file_status = std::filesystem::status(path.parent_path(), ec);
        if (!ec) {
          auto perms = file_status.permissions();
          HOLOSCAN_LOG_ERROR("Parent directory permissions: {}", static_cast<int>(perms));
        }
      }

      throw std::runtime_error("Failed to open output file: " + file_path);
    }

    HOLOSCAN_LOG_INFO("Successfully opened output file: {}", file_path);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext&,
               ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) { throw std::runtime_error("Failed to receive input"); }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    if (frames_recorded_ > num_of_frames_to_record_.get()) { return; }

    auto maybe_video_buffer = entity.get<nvidia::gxf::VideoBuffer>();
    if (maybe_video_buffer) {
      process_video_buffer(maybe_video_buffer.value());
    } else {
      auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
      if (!maybe_tensor) {
        HOLOSCAN_LOG_ERROR("Neither VideoBuffer nor Tensor found in message");
        return;
      }
      process_gxf_tensor(maybe_tensor.value());
    }

    frames_recorded_++;
  }

 private:
  /**
   * @brief Processes a VideoBuffer input and writes its data to file.
   *
   * Extracts data from the VideoBuffer, determines memory storage type,
   * and calls write_data_to_file to handle the actual file writing.
   *
   * @param video_buffer Handle to the GXF VideoBuffer to process
   */
  void process_video_buffer(nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> video_buffer) {
    const auto buffer_size = video_buffer->size();
    const auto storage_type = video_buffer->storage_type();
    const auto data_ptr = video_buffer->pointer();

    HOLOSCAN_LOG_TRACE("Processing VideoBuffer: size={}, storage_type={}",
                       buffer_size,
                       static_cast<int>(storage_type));

    write_data_to_file(data_ptr, buffer_size, storage_type);
  }

  /**
   * @brief Processes a GXF Tensor input and writes its data to file.
   *
   * Extracts data from the GXF Tensor, determines memory storage type,
   * and calls write_data_to_file to handle the actual file writing.
   *
   * @param tensor Handle to the GXF Tensor to process
   */
  void process_gxf_tensor(nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor) {
    const auto tensor_size = tensor->size();
    const auto storage_type = tensor->storage_type();
    const auto data_ptr = tensor->pointer();

    HOLOSCAN_LOG_TRACE(
        "Processing Tensor: size={}, storage_type={}", tensor_size, static_cast<int>(storage_type));

    write_data_to_file(data_ptr, tensor_size, storage_type);
  }

  /**
   * @brief Writes data to file, handling both host and device memory sources.
   *
   * For host memory (kHost, kSystem), writes data directly to file.
   * For device memory (kDevice), copies data to host buffer first, then writes to file.
   * Automatically resizes the host buffer as needed and includes comprehensive error checking.
   *
   * @param data_ptr Pointer to the data to write
   * @param data_size Size of the data in bytes
   * @param storage_type Memory storage type indicating where the data resides
   *
   * @throws std::runtime_error If file stream is in bad state, CUDA operations fail,
   *                           or unsupported memory storage type is encountered
   */
  void write_data_to_file(void* data_ptr, size_t data_size,
                          nvidia::gxf::MemoryStorageType storage_type) {
    if (!data_ptr || data_size == 0) {
      HOLOSCAN_LOG_ERROR(
          "Invalid data pointer or size: ptr={}, size={}", static_cast<void*>(data_ptr), data_size);
      return;
    }

    if (!file_stream_.is_open() || !file_stream_.good()) {
      HOLOSCAN_LOG_ERROR("File stream is not open or in bad state");
      throw std::runtime_error("File stream error");
    }

    // Ensure host buffer is large enough
    if (host_buffer_.size() < data_size) {
      HOLOSCAN_LOG_TRACE(
          "Resizing host buffer from {} to {} bytes", host_buffer_.size(), data_size);
      host_buffer_.resize(data_size);
    }

    switch (storage_type) {
      case nvidia::gxf::MemoryStorageType::kHost:
      case nvidia::gxf::MemoryStorageType::kSystem: {
        // Data is already on host, write directly
        file_stream_.write(reinterpret_cast<const char*>(data_ptr), data_size);
        break;
      }
      case nvidia::gxf::MemoryStorageType::kDevice: {
        // Data is on device, copy to host first
        CUDA_TRY(cudaMemcpy(host_buffer_.data(), data_ptr, data_size, cudaMemcpyDeviceToHost));
        file_stream_.write(reinterpret_cast<const char*>(host_buffer_.data()), data_size);
        break;
      }
      default: {
        HOLOSCAN_LOG_ERROR("Unsupported memory storage type: {}", static_cast<int>(storage_type));
        throw std::runtime_error("Unsupported memory storage type");
      }
    }

    if (!file_stream_.good()) {
      HOLOSCAN_LOG_ERROR("Failed to write data to file - stream state: fail={}, bad={}, eof={}",
                         file_stream_.fail(),
                         file_stream_.bad(),
                         file_stream_.eof());
      throw std::runtime_error("Failed to write data to file");
    }

    // Flush to ensure data is written
    file_stream_.flush();

    HOLOSCAN_LOG_TRACE(
        "Successfully wrote {} bytes to file (frame {})", data_size, frames_recorded_ + 1);
  }

  std::ofstream file_stream_;
  std::vector<uint8_t> host_buffer_;  // Buffer for device-to-host copies
  uint32_t frames_recorded_ = 0;
  Parameter<uint32_t> num_of_frames_to_record_;
  Parameter<std::string> file_path_;
};

/**
 * @class MockReceiverOp
 * @brief A mock operator to simulate data reception when no output operator is defined.
 */
class MockReceiverOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MockReceiverOp)

  MockReceiverOp() = default;

  void initialize() override {
    cudaError_t cuda_error;
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::initialize()");
    holoscan::Operator::initialize();

    port_id_ = get_port_id(interface_name_.get());
    if (port_id_ == -1) {
      HOLOSCAN_LOG_ERROR("Invalid RX port {} specified in the config", interface_name_.get());
      exit(1);
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(interface_name_,
                            "interface_name",
                            "Port name",
                            "Name of the port to poll on from the advanced_network config",
                            "rx_port");
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {

    BurstParams *burst;

    // In this example, we'll loop through all the rx queues of the interface
    // assuming we want to process the packets the same way for all queues
    const auto num_rx_queues = get_num_rx_queues(port_id_);
    for (int q = 0; q < num_rx_queues; q++) {
      auto status = get_rx_burst(&burst, port_id_, q);

      if (status != Status::SUCCESS) {
        HOLOSCAN_LOG_DEBUG("No RX burst available");
        continue;
      }

      free_all_packets_and_burst_rx(burst);
    }
  }

 private:
  int port_id_ = 0;
  Parameter<std::string> interface_name_;                // Port name from advanced_network config
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
    if (advanced_network::adv_net_init(adv_net_config) != advanced_network::Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
      exit(1);
    }
    HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

    const auto [rx_en, tx_en] = advanced_network::get_rx_tx_configs_enabled(config());
    const auto mgr_type = advanced_network::get_manager_type(config());

    HOLOSCAN_LOG_INFO("Using Advanced Network manager {}",
                      advanced_network::manager_type_to_string(mgr_type));

    HOLOSCAN_LOG_INFO("Using ANO manager {}", advanced_network::manager_type_to_string(mgr_type));
    if (!rx_en) {
      HOLOSCAN_LOG_ERROR("Rx is not enabled. Please enable Rx in the config file.");
      exit(1);
    }

    std::string interface_name = "";

    auto multi_streams_adv_net_media_rx_yaml = config().yaml_nodes()[0]["advanced_network_media_rx"];
    std::unordered_map<std::string, std::shared_ptr<ops::AdvNetworkMediaRxOp>> adv_net_media_rx_map;
    for (const auto& stream : multi_streams_adv_net_media_rx_yaml) {
      std::string stream_name = stream["name"].as<std::string>();
      interface_name = stream["interface_name"].as<std::string>("");
      auto adv_net_media_rx = make_operator<ops::AdvNetworkMediaRxOp>(
          "adv_net_media_rx_" + stream_name,
          Arg("interface_name", stream["interface_name"].as<std::string>("")),
          Arg("queue_id", stream["queue_id"].as<uint16_t>(0)),
          Arg("stream_id", stream["stream_id"].as<uint32_t>(0)),
          Arg("frame_width", stream["frame_width"].as<uint32_t>(1920)),
          Arg("frame_height", stream["frame_height"].as<uint32_t>(1080)),
          Arg("bit_depth", stream["bit_depth"].as<uint32_t>(8)),
          Arg("video_format", stream["video_format"].as<std::string>("RGB888")),
          Arg("hds", stream["hds"].as<bool>(true)),
          Arg("output_format", stream["output_format"].as<std::string>("video_buffer")),
          Arg("memory_location", stream["memory_location"].as<std::string>("device")),
          make_condition<BooleanCondition>("is_alive", true)
      );
      adv_net_media_rx_map[stream_name] = adv_net_media_rx;
    }
    if (adv_net_media_rx_map.empty()) {
      HOLOSCAN_LOG_ERROR("No advanced_network_media_rx entries found in the config");
      exit(1);
    }

    const auto allocator = make_resource<holoscan::UnboundedAllocator>("allocator");

    if (from_config("media_player_config.visualize").as<bool>()) {
      if (multi_streams_adv_net_media_rx_yaml.size() > 1) {
        HOLOSCAN_LOG_ERROR("Visualization with multiple streams is not supported yet.");
        exit(1);
      }
      const auto cuda_stream_pool =
          make_resource<holoscan::CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

      auto visualizer = make_operator<ops::HolovizOp>("visualizer",
                                                      from_config("holoviz"),
                                                      Arg("cuda_stream_pool", cuda_stream_pool),
                                                      Arg("allocator") = allocator);
      add_flow(adv_net_media_rx_map[0], visualizer, {{"out_video_buffer", "receivers"}});
    } else if (from_config("media_player_config.write_to_file").as<bool>()) {
      auto frames_writer_yaml = config().yaml_nodes()[0]["frames_writer"];
      if (frames_writer_yaml.size() == adv_net_media_rx_map.size()) {
        for (const auto& stream : frames_writer_yaml) {
          std::string stream_name = stream["name"].as<std::string>();
          auto frames_writer = make_operator<ops::FramesWriterOp>(
            "frames_writer_" + stream_name,
            Arg("file_path", stream["file_path"].as<std::string>("")),
            Arg("num_of_frames_to_record", stream["num_of_frames_to_record"].as<uint32_t>(1000))
          );
          // Connect the corresponding network rx operator to this frames writer
          if (adv_net_media_rx_map.find(stream_name) != adv_net_media_rx_map.end()) {
            add_flow(adv_net_media_rx_map[stream_name], frames_writer);
          } else {
            HOLOSCAN_LOG_ERROR("Stream {} not found in adv_net_media_rx_map", stream_name);
            exit(1);
          }
        }
      } else {
        HOLOSCAN_LOG_ERROR("Number of frames_writer entries must match number of advanced_network_media_rx entries");
        exit(1);
      }
    } else {
      HOLOSCAN_LOG_WARN("No output type (write_to_file/visualize) defined. Data will be received but not processed.");
      auto mock_receiver = make_operator<ops::MockReceiverOp>(
          "mock_receiver",
          Arg("interface_name", interface_name),
          make_condition<BooleanCondition>("is_alive", true)
      );
      add_operator(mock_receiver);
    }
  }
};

int main(int argc, char** argv) {
  using namespace holoscan;
  auto app = holoscan::make_application<App>();

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
    return -1;
  }

  std::filesystem::path config_path(argv[1]);
  if (!config_path.is_absolute()) {
    config_path = std::filesystem::canonical(argv[0]).parent_path() / config_path;
  }

  app->config(config_path);
  app->scheduler(app->make_scheduler<MultiThreadScheduler>("multithread-scheduler",
                                                           app->from_config("scheduler")));
  app->run();

  advanced_network::shutdown();

  return 0;
}
