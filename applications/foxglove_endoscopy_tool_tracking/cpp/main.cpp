/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <getopt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <fmt/format.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "foxglove_publisher.hpp"

namespace {

constexpr uint32_t kEndoscopyWidth = 854;
constexpr uint32_t kEndoscopyHeight = 480;
constexpr uint32_t kToolMaskWidth = 107;
constexpr uint32_t kToolMaskHeight = 60;

void throw_on_cuda_error(cudaError_t error, const std::string& action) {
  if (error != cudaSuccess) {
    throw std::runtime_error(fmt::format("{} failed: {}", action, cudaGetErrorString(error)));
  }
}

std::vector<float> copy_float_tensor(const holoscan::Tensor& tensor, cudaStream_t stream) {
  if (tensor.dtype().code != kDLFloat || tensor.dtype().bits != 32 || tensor.dtype().lanes != 1) {
    throw std::runtime_error("Tool tracking Foxglove adapter expects float32 tensors");
  }

  std::vector<float> values(tensor.nbytes() / sizeof(float));
  const auto device = tensor.device();
  const bool is_device = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
  if (is_device) {
    throw_on_cuda_error(cudaMemcpyAsync(values.data(),
                                        tensor.data(),
                                        tensor.nbytes(),
                                        cudaMemcpyDeviceToHost,
                                        stream),
                        "cudaMemcpyAsyncDeviceToHost");
    throw_on_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  } else {
    std::memcpy(values.data(), tensor.data(), tensor.nbytes());
  }
  return values;
}

uint64_t timestamp_from_input(holoscan::Operator& op,
                              holoscan::InputContext& op_input,
                              const char* port_name) {
  if (op.is_metadata_enabled()) {
    const auto meta = op.metadata();
    if (meta) {
      for (const auto& key : {"acquisition_timestamp_ns", "timestamp_ns", "sensor_timestamp_ns"}) {
        if (meta->has_key(key)) {
          const auto timestamp = meta->get<uint64_t>(key, 0);
          if (timestamp > 0) {
            return timestamp;
          }
        }
      }
    }
  }

  if (const auto timestamp = op_input.get_acquisition_timestamp(port_name)) {
    if (timestamp.value() > 0) {
      return static_cast<uint64_t>(timestamp.value());
    }
  }
  return holoscan::ops::now_epoch_ns();
}

std::optional<std::pair<double, double>> image_coordinates(double x, double y) {
  if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 || y < 0.0) {
    return std::nullopt;
  }
  if (x <= 1.0 && y <= 1.0) {
    return std::make_pair(x * static_cast<double>(kEndoscopyWidth),
                          y * static_cast<double>(kEndoscopyHeight));
  }
  if (x <= kToolMaskWidth && y <= kToolMaskHeight) {
    return std::make_pair(x * static_cast<double>(kEndoscopyWidth) /
                              static_cast<double>(kToolMaskWidth),
                          y * static_cast<double>(kEndoscopyHeight) /
                              static_cast<double>(kToolMaskHeight));
  }
  if (x <= kEndoscopyWidth && y <= kEndoscopyHeight) {
    return std::make_pair(x, y);
  }
  return std::nullopt;
}

class ToolTrackingFoxgloveAdapterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ToolTrackingFoxgloveAdapterOp)

  ToolTrackingFoxgloveAdapterOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<std::shared_ptr<holoscan::ops::FoxgloveBatch>>("messages");
    spec.param(annotation_topic_,
               "annotation_topic",
               "Annotation topic",
               "Foxglove ImageAnnotations topic for tracked tools",
               std::string("/detections"));
    spec.param(state_topic_,
               "state_topic",
               "State topic",
               "Foxglove KeyValuePair topic for inference FPS",
               std::string("/state/inference_fps"));
    spec.param(labels_,
               "labels",
               "Labels",
               "Tool labels in model output order",
               std::vector<std::string>{"Grasper",
                                        "Bipolar",
                                        "Hook",
                                        "Scissors",
                                        "Clipper",
                                        "Irrigator",
                                        "Spec.Bag"});
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
      return;
    }

    auto tensor = maybe_entity.value().get<holoscan::Tensor>("scaled_coords");
    if (!tensor) {
      throw std::runtime_error("Tool tracking output is missing tensor 'scaled_coords'");
    }

    const auto stream = op_input.receive_cuda_stream("input", false, true);
    const auto coords = copy_float_tensor(*tensor, stream);
    if (coords.size() % 3 != 0) {
      throw std::runtime_error("Tool tracking tensor 'scaled_coords' must be Nx3");
    }

    auto batch = std::make_shared<holoscan::ops::FoxgloveBatch>();
    holoscan::ops::FoxgloveImageAnnotations annotations;
    annotations.topic = annotation_topic_.get();
    annotations.timestamp_ns = timestamp_from_input(*this, op_input, "input");

    const auto labels = labels_.get();
    for (size_t index = 0; index < coords.size() / 3; ++index) {
      const auto maybe_point = image_coordinates(coords[index * 3 + 0], coords[index * 3 + 1]);
      if (!maybe_point) {
        continue;
      }
      const auto [x, y] = maybe_point.value();
      const auto marker_size = static_cast<double>(coords[index * 3 + 2]);
      if (!std::isfinite(marker_size) || marker_size <= 0.0) {
        continue;
      }

      const auto label = index < labels.size() ? labels[index] : fmt::format("tool_{}", index);
      const double box_extent =
          marker_size <= 1.0
              ? marker_size * static_cast<double>(std::min(kEndoscopyWidth, kEndoscopyHeight))
              : marker_size;
      holoscan::ops::FoxgloveBox2D box;
      box.x = std::clamp(x - box_extent * 0.5, 0.0, static_cast<double>(kEndoscopyWidth - 1));
      box.y = std::clamp(y - box_extent * 0.5, 0.0, static_cast<double>(kEndoscopyHeight - 1));
      box.width = std::min(box_extent, static_cast<double>(kEndoscopyWidth) - box.x);
      box.height = std::min(box_extent, static_cast<double>(kEndoscopyHeight) - box.y);
      box.label = label;
      annotations.boxes.push_back(std::move(box));

      holoscan::ops::FoxglovePointsAnnotation point_set;
      point_set.type = foxglove::messages::PointsAnnotation::PointsAnnotationType::POINTS;
      point_set.label = label;
      point_set.thickness = std::max(4.0, box_extent * 0.4);
      point_set.points.push_back({x, y, -1.0, label});
      annotations.point_sets.push_back(std::move(point_set));

      holoscan::ops::FoxgloveText text;
      text.x = x + 6.0;
      text.y = y - 6.0;
      text.text = label;
      text.font_size = 14.0;
      annotations.texts.push_back(std::move(text));
    }

    batch->annotations.push_back(std::move(annotations));

    const auto now = std::chrono::steady_clock::now();
    if (last_tick_.time_since_epoch().count() != 0) {
      const auto elapsed = std::chrono::duration<double>(now - last_tick_).count();
      if (elapsed > 0.0) {
        holoscan::ops::FoxgloveKeyValue fps;
        fps.topic = state_topic_.get();
        fps.key = "inference_fps";
        fps.value = fmt::format("{:.2f}", 1.0 / elapsed);
        fps.timestamp_ns = holoscan::ops::now_epoch_ns();
        batch->key_values.push_back(std::move(fps));
      }
    }
    last_tick_ = now;

    op_output.emit(batch, "messages");
  }

 private:
  holoscan::Parameter<std::string> annotation_topic_;
  holoscan::Parameter<std::string> state_topic_;
  holoscan::Parameter<std::vector<std::string>> labels_;
  std::chrono::steady_clock::time_point last_tick_{};
};

class App : public holoscan::Application {
 public:
  void set_datapath(std::string path) { datapath_ = std::move(path); }

  void compose() override {
    using namespace holoscan;

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto replayer = make_operator<ops::VideoStreamReplayerOp>(
        "replayer", from_config("replayer"), Arg("directory", datapath_));
    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config("format_converter"),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, kEndoscopyWidth * kEndoscopyHeight * 3 * 4, 2),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = datapath_ + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = datapath_ + "/engines";
    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, 107 * 60 * 7 * 4, 2 + 5 * 2),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    auto postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        Arg("device_allocator") = make_resource<BlockMemoryPool>(
            "device_allocator", 1, 107 * 60 * 7 * 4, 4));

    auto tracking_adapter = make_operator<ToolTrackingFoxgloveAdapterOp>(
        "tool_tracking_foxglove", from_config("tool_tracking_foxglove"));

    auto foxglove = make_operator<ops::FoxglovePublisherOp>(
        "foxglove",
        from_config("foxglove"),
        Arg("image_topic", std::string("/video")),
        Arg("image_frame_id", std::string("endoscope")));

    add_flow(replayer, foxglove, {{"output", "image"}});
    add_flow(replayer, format_converter, {{"output", "source_video"}});
    add_flow(format_converter, lstm_inferer);
    add_flow(lstm_inferer, postprocessor, {{"tensor", "in"}});
    add_flow(postprocessor, tracking_adapter, {{"out", "input"}});
    add_flow(tracking_adapter, foxglove, {{"messages", "messages"}});
  }

 private:
  std::string datapath_;
};

bool parse_arguments(int argc, char** argv, std::string& data_path, std::string& config_path) {
  static struct option long_options[] = {
      {"data", required_argument, nullptr, 'd'},
      {"config", required_argument, nullptr, 'c'},
      {nullptr, 0, nullptr, 0}};

  while (true) {
    const int c = getopt_long(argc, argv, "d:c:", long_options, nullptr);
    if (c == -1) {
      break;
    }
    switch (c) {
      case 'c':
        config_path = optarg;
        break;
      case 'd':
        data_path = optarg;
        break;
      default:
        return false;
    }
  }
  return true;
}

std::filesystem::path executable_directory(const char* executable_path) {
  try {
    return std::filesystem::canonical(executable_path).parent_path();
  } catch (const std::filesystem::filesystem_error&) {
    return std::filesystem::current_path();
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path;
  std::string data_directory;
  if (!parse_arguments(argc, argv, data_directory, config_path)) {
    return 1;
  }

  if (data_directory.empty()) {
    if (const auto* input_path = std::getenv("HOLOSCAN_INPUT_PATH");
        input_path != nullptr && input_path[0] != '\0') {
      data_directory = input_path;
    } else if (std::filesystem::is_directory(std::filesystem::current_path() / "data/endoscopy")) {
      data_directory = (std::filesystem::current_path() / "data/endoscopy").string();
    } else {
      HOLOSCAN_LOG_ERROR("Input data not provided. Use --data or HOLOSCAN_INPUT_PATH.");
      return 1;
    }
  }

  if (config_path.empty()) {
    if (const auto* config_env = std::getenv("HOLOSCAN_CONFIG_PATH");
        config_env != nullptr && config_env[0] != '\0') {
      config_path = config_env;
    } else {
      config_path =
          (executable_directory(argv[0]) / "foxglove_endoscopy_tool_tracking.yaml").string();
    }
  }

  auto app = holoscan::make_application<App>();
  app->enable_metadata(true);
  app->config(config_path);
  app->set_datapath(data_directory);
  app->run();
  return 0;
}
