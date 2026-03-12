/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

// ============================================================================
// IMX274 GPU-Resident Application
//
//   - hsb_roce_receiver_nmd   (RoceReceiverOp)
//   - csi_to_bayer_gpu_resident (CsiToBayerGpuResidentOp)
//   - image_processor_gpu_resident (ImageProcessorGpuResidentOp)
//   - display_gpu_resident (DisplayGpuResidentOp)
//   - BayerDemosaicGpuResidentOp (from Holoscan SDK)
// ============================================================================

#include <getopt.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

// Holohub operators
#include <csi_to_bayer_gpu_resident/csi_to_bayer_converter_base.hpp>
#include <csi_to_bayer_gpu_resident/csi_to_bayer_gpu_resident.hpp>
#include <display_gpu_resident/display_gpu_resident.hpp>
#include <hsb_roce_receiver_nmd/roce_receiver_op.hpp>
#include <image_processor_gpu_resident/image_processor_gpu_resident.hpp>

// Holoscan SDK
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic_gpu_resident/bayer_demosaic_gpu_resident.hpp>

#include <fmt/format.h>
#include <holoscan/logger/logger.hpp>

// Application-local operators
#include "data_ready_input_op.hpp"
#include "frame_source_op_gr.hpp"
#include "noop_sink_op.hpp"
#include "shared_frame_state.hpp"

using namespace imx274_gpu_resident;

namespace {

// ============================================================================
// Fragments
// ============================================================================

class DataReadyGpuResidentFragment : public holoscan::Fragment {
 public:
  void compose() override {
    auto input_op = make_operator<DataReadyInputOp>("input_op");
    add_operator(input_op);
  }
};

class ReceiverFragment : public holoscan::Fragment {
 public:
  void configure(size_t frame_size, CUcontext cuda_context, const std::string& ibv_name,
                 uint32_t ibv_port, hololink::DataChannel& hololink_channel,
                 std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera,
                 std::shared_ptr<SharedFrameState> shared_state) {
    frame_size_ = frame_size;
    cuda_context_ = cuda_context;
    ibv_name_ = ibv_name;
    ibv_port_ = ibv_port;
    hololink_channel_ = &hololink_channel;
    camera_ = std::move(camera);
    shared_state_ = std::move(shared_state);
  }

  void compose() override {
    receiver_tick_condition_ =
        make_condition<holoscan::BooleanCondition>("receiver_tick", true);
    auto receiver = make_operator<hololink::operators::RoceReceiverOp>(
        "receiver",
        receiver_tick_condition_,
        holoscan::Arg("frame_size", frame_size_),
        holoscan::Arg("frame_context", cuda_context_),
        holoscan::Arg("ibv_name", ibv_name_),
        holoscan::Arg("ibv_port", ibv_port_),
        holoscan::Arg("hololink_channel", hololink_channel_),
        holoscan::Arg("skip_host_metadata", true),
        holoscan::Arg("device_start", std::function<void()>([this] { camera_->start(); })),
        holoscan::Arg("device_stop", std::function<void()>([this] { camera_->stop(); })));

    {
      std::lock_guard<std::mutex> lock(shared_state_->mutex);
      shared_state_->get_frame_memory_base = [receiver]() { return receiver->frame_memory_base(); };
      shared_state_->frame_size_rounded =
          hololink::core::round_up(frame_size_, hololink::core::PAGE_SIZE);
    }

    auto sink = make_operator<NoopSinkOp>("noop_sink");
    add_flow(receiver, sink, {{"output", "input"}});
  }

  std::shared_ptr<holoscan::BooleanCondition> receiver_tick_condition() const {
    return receiver_tick_condition_;
  }

 private:
  size_t frame_size_ = 0;
  CUcontext cuda_context_ = nullptr;
  std::string ibv_name_;
  uint32_t ibv_port_ = 0;
  hololink::DataChannel* hololink_channel_ = nullptr;
  std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera_;
  std::shared_ptr<SharedFrameState> shared_state_;
  std::shared_ptr<holoscan::BooleanCondition> receiver_tick_condition_;
};

class Imx274GpuResidentFragment : public holoscan::Fragment {
 public:
  void configure(size_t frame_size, uint32_t width, uint32_t height,
                 hololink::csi::PixelFormat pixel_format, hololink::csi::BayerFormat bayer_format,
                 std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera,
                 std::shared_ptr<SharedFrameState> shared_state, bool gsync,
                 bool front_buffer_rendering, int32_t refresh_rate_hz,
                 int32_t display_width, int32_t display_height) {
    frame_size_ = frame_size;
    width_ = width;
    height_ = height;
    pixel_format_ = pixel_format;
    bayer_format_ = bayer_format;
    camera_ = std::move(camera);
    shared_state_ = std::move(shared_state);
    gsync_ = gsync;
    front_buffer_rendering_ = front_buffer_rendering;
    refresh_rate_hz_ = refresh_rate_hz;
    display_width_ = display_width;
    display_height_ = display_height;
  }

  void compose() override {
    auto source_op =
        make_operator<FrameSourceOpGR>("source_op", holoscan::Arg("output_size", frame_size_));
    source_op->set_shared_state(shared_state_);

    // CSI-to-Bayer GPU-resident operator (holohub)
    auto csi_to_bayer =
        make_operator<hololink::operators::CsiToBayerGpuResidentOp>("csi_to_bayer_gr");
    camera_->configure_converter(csi_to_bayer);

    // Image processor GPU-resident operator (holohub)
    auto image_processor = make_operator<hololink::operators::ImageProcessorGpuResidentOp>(
        "image_processor_gr",
        holoscan::Arg("pixel_format", int(pixel_format_)),
        holoscan::Arg("bayer_format", int(bayer_format_)),
        holoscan::Arg("optical_black", 0),
        holoscan::Arg("width", static_cast<int32_t>(width_)),
        holoscan::Arg("height", static_cast<int32_t>(height_)));

    // Bayer demosaic GPU-resident operator (Holoscan SDK)
    auto demosaic = make_operator<holoscan::ops::BayerDemosaicGpuResidentOp>(
        "demosaic_gr",
        holoscan::Arg("width", static_cast<int32_t>(width_)),
        holoscan::Arg("height", static_cast<int32_t>(height_)),
        holoscan::Arg("pixel_type", 1),
        holoscan::Arg("generate_alpha", true),
        holoscan::Arg("alpha_value", 65535),
        holoscan::Arg("bayer_grid_pos", int(bayer_format_)),
        holoscan::Arg("interpolation_mode", 0));

    // Display GPU-resident operator (holohub)
    auto display = make_operator<holoscan::ops::DisplayGpuResidentOp>(
        "display_gr",
        holoscan::Arg("width", static_cast<int32_t>(width_)),
        holoscan::Arg("height", static_cast<int32_t>(height_)),
        holoscan::Arg("out_channels", 4),
        holoscan::Arg("element_size", 2),
        holoscan::Arg("display_width", display_width_),
        holoscan::Arg("display_height", display_height_),
        holoscan::Arg("surface_format",
                      static_cast<int32_t>(
                          holoscan::ops::DisplayOpSurfaceFormat::kDisplayOpSurfaceFormatA8R8G8B8)),
        holoscan::Arg("front_buffer_rendering", front_buffer_rendering_),
        holoscan::Arg("gsync", gsync_),
        // for gsync, we use the maximum available refresh rate
        holoscan::Arg("refresh_rate", gsync_ ? 0 : refresh_rate_hz_ * 1000));

    // Pipeline: source -> csi_to_bayer -> image_processor -> demosaic -> display
    add_flow(source_op, csi_to_bayer);
    add_flow(csi_to_bayer, image_processor);
    add_flow(image_processor, demosaic);
    add_flow(demosaic, display);

    source_op_ = source_op;
  }

  void set_receiver_tick_condition(std::shared_ptr<holoscan::BooleanCondition> tick_condition) {
    receiver_tick_condition_ = std::move(tick_condition);
  }

  std::shared_ptr<holoscan::BooleanCondition> receiver_tick_condition() const {
    return receiver_tick_condition_;
  }

  void stop_execution(const std::string& op_name = "") override {
    if (receiver_tick_condition_) {
      receiver_tick_condition_->disable_tick();
      HOLOSCAN_LOG_INFO("Disabled receiver tick condition");
    }
    holoscan::Fragment::stop_execution(op_name);
  }

  std::shared_ptr<holoscan::GPUResidentOperator> source_op() const { return source_op_; }
  std::shared_ptr<SharedFrameState> shared_state() const { return shared_state_; }

 private:
  size_t frame_size_ = 0;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  hololink::csi::PixelFormat pixel_format_ = hololink::csi::PixelFormat::RAW_10;
  hololink::csi::BayerFormat bayer_format_ = hololink::csi::BayerFormat::RGGB;
  bool gsync_ = false;
  bool front_buffer_rendering_ = true;
  int32_t refresh_rate_hz_ = 60;
  int32_t display_width_ = 2560;
  int32_t display_height_ = 1440;
  std::shared_ptr<holoscan::BooleanCondition> receiver_tick_condition_;
  std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera_;
  std::shared_ptr<SharedFrameState> shared_state_;
  std::shared_ptr<holoscan::GPUResidentOperator> source_op_;
};

// ============================================================================
// Application
// ============================================================================

struct AppConfig {
  CUcontext cuda_context;
  hololink::DataChannel* hololink_channel;
  std::string ibv_name;
  uint32_t ibv_port;
  std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera;
  hololink::sensors::imx274_mode::Mode camera_mode;
  size_t frame_size;
  uint32_t width;
  uint32_t height;
  hololink::csi::PixelFormat pixel_format;
  hololink::csi::BayerFormat bayer_format;
  bool gsync;
  bool front_buffer_rendering;
  int32_t refresh_rate_hz;
  int32_t display_width;
  int32_t display_height;
  std::shared_ptr<SharedFrameState> shared_state;
};

class Imx274GrApplication : public holoscan::Application {
 public:
  explicit Imx274GrApplication(AppConfig config) : config_(std::move(config)) {}

  void compose() override {
    config_.camera->set_mode(config_.camera_mode);

    auto receiver_fragment_base = make_fragment<ReceiverFragment>("receiver_fragment");
    auto receiver_fragment = std::dynamic_pointer_cast<ReceiverFragment>(receiver_fragment_base);
    receiver_fragment->configure(config_.frame_size,
                                 config_.cuda_context,
                                 config_.ibv_name,
                                 config_.ibv_port,
                                 *config_.hololink_channel,
                                 config_.camera,
                                 config_.shared_state);
    receiver_fragment->compose_graph();

    auto gr_fragment_base = make_fragment<Imx274GpuResidentFragment>("gr_fragment");
    auto gr_fragment = std::dynamic_pointer_cast<Imx274GpuResidentFragment>(gr_fragment_base);
    gr_fragment->configure(config_.frame_size,
                           config_.width,
                           config_.height,
                           config_.pixel_format,
                           config_.bayer_format,
                           config_.camera,
                           config_.shared_state,
                           config_.gsync,
                           config_.front_buffer_rendering,
                           config_.refresh_rate_hz,
                           config_.display_width,
                           config_.display_height);
    gr_fragment->set_receiver_tick_condition(receiver_fragment->receiver_tick_condition());

    gr_fragment->compose_graph();

    setup_data_ready_handler(gr_fragment);

    add_fragment(receiver_fragment_base);
    add_fragment(gr_fragment_base);
  }

 private:
  void setup_data_ready_handler(const std::shared_ptr<Imx274GpuResidentFragment>& gr_fragment) {
    auto data_ready_fragment = make_fragment<DataReadyGpuResidentFragment>("data_ready_fragment");
    data_ready_fragment->compose_graph();

    auto data_ready_graph = data_ready_fragment->graph_shared();
    auto input_node = data_ready_graph->find_node("input_op");
    if (!input_node) {
      throw std::runtime_error("Could not find input_op in data ready handler fragment");
    }

    auto* input_op = dynamic_cast<DataReadyInputOp*>(input_node.get());
    if (!input_op) {
      throw std::runtime_error("Could not cast input_op to DataReadyInputOp");
    }

    input_op->set_shared_state(gr_fragment->shared_state());

    gr_fragment->gpu_resident().register_data_ready_handler(std::move(data_ready_fragment));
  }

  AppConfig config_;
};

}  // anonymous namespace

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
  auto camera_mode = hololink::sensors::imx274_mode::IMX274_MODE_1920X1080_60FPS;
  std::string configuration;
  std::string hololink_ip = "192.168.0.2";
  holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
  uint32_t ibv_port = 1;
  int32_t expander_configuration = 0;
  int32_t pattern = 0;
  bool pattern_set = false;
  bool gsync = false;
  int32_t refresh_rate_hz = 60;
  int32_t display_width = 2560;
  int32_t display_height = 1440;
  enum class DisplayModeOverride { None, Gsync, Fbr };
  DisplayModeOverride display_mode_override = DisplayModeOverride::None;

  std::string ibv_name("roceP5p3s0f0");
  try {
    ibv_name = hololink::infiniband_devices()[0];
  } catch (const std::exception& e) {
    std::cerr << "Error getting IBV name: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  const struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                        {"camera-mode", required_argument, nullptr, 0},
                                        {"configuration", required_argument, nullptr, 0},
                                        {"hololink", required_argument, nullptr, 0},
                                        {"ibv-name", required_argument, nullptr, 0},
                                        {"ibv-port", required_argument, nullptr, 0},
                                        {"expander-configuration", required_argument, nullptr, 0},
                                        {"pattern", required_argument, nullptr, 0},
                                        {"gsync", no_argument, nullptr, 0},
                                        {"fbr", no_argument, nullptr, 0},
                                        {"refresh", required_argument, nullptr, 'r'},
                                        {"resolution", required_argument, nullptr, 'l'},
                                        {"log-level", required_argument, nullptr, 0},
                                        {0, 0, nullptr, 0}};
  try {
    while (true) {
      int option_index = 0;
      const int c = getopt_long(argc, argv, "hl:r:", long_options, &option_index);

      if (c == -1) {
        break;
      }

      const std::string argument(optarg ? optarg : "");
      if (c == 0) {
        const struct option* cur_option = &long_options[option_index];
        if (cur_option->name == std::string("camera-mode")) {
          camera_mode = static_cast<hololink::sensors::imx274_mode::Mode>(std::stoi(argument));
        } else if (cur_option->name == std::string("configuration")) {
          configuration = argument;
        } else if (cur_option->name == std::string("hololink")) {
          hololink_ip = argument;
        } else if (cur_option->name == std::string("log-level")) {
          if ((argument == "trace") || (argument == "TRACE")) {
            log_level = holoscan::LogLevel::TRACE;
          } else if ((argument == "debug") || (argument == "DEBUG")) {
            log_level = holoscan::LogLevel::DEBUG;
          } else if ((argument == "info") || (argument == "INFO")) {
            log_level = holoscan::LogLevel::INFO;
          } else if ((argument == "warn") || (argument == "WARN")) {
            log_level = holoscan::LogLevel::WARN;
          } else if ((argument == "error") || (argument == "ERROR")) {
            log_level = holoscan::LogLevel::ERROR;
          } else if ((argument == "critical") || (argument == "CRITICAL")) {
            log_level = holoscan::LogLevel::CRITICAL;
          } else if ((argument == "off") || (argument == "OFF")) {
            log_level = holoscan::LogLevel::OFF;
          } else {
            throw std::runtime_error(fmt::format("Unhandled log level \"{}\"", argument));
          }
        } else if (cur_option->name == std::string("ibv-name")) {
          ibv_name = argument;
        } else if (cur_option->name == std::string("ibv-port")) {
          ibv_port = std::stoul(argument);
        } else if (cur_option->name == std::string("expander-configuration")) {
          expander_configuration = std::stoul(argument);
        } else if (cur_option->name == std::string("pattern")) {
          pattern = std::stoi(argument);
          pattern_set = true;
        } else if (cur_option->name == std::string("gsync")) {
          if (display_mode_override == DisplayModeOverride::Fbr) {
            throw std::runtime_error("Options --gsync and --fbr are mutually exclusive.");
          }
          display_mode_override = DisplayModeOverride::Gsync;
          gsync = true;
        } else if (cur_option->name == std::string("fbr")) {
          if (display_mode_override == DisplayModeOverride::Gsync) {
            throw std::runtime_error("Options --gsync and --fbr are mutually exclusive.");
          }
          display_mode_override = DisplayModeOverride::Fbr;
          gsync = false;
        } else {
          throw std::runtime_error(fmt::format("Unhandled option \"{}\"", cur_option->name));
        }
      } else {
        switch (c) {
          case 'l': {
            const std::string res_arg(optarg ? optarg : "");
            auto x_pos = res_arg.find('x');
            if (x_pos == std::string::npos) {
              throw std::runtime_error(
                  fmt::format("Invalid resolution format \"{}\", expected WxH (e.g. 1920x1080)",
                              res_arg));
            }
            display_width = std::stoi(res_arg.substr(0, x_pos));
            display_height = std::stoi(res_arg.substr(x_pos + 1));
            if (display_width <= 0 || display_height <= 0) {
              throw std::runtime_error("Resolution width and height must be positive.");
            }
            break;
          }
          case 'r':
            refresh_rate_hz = std::stoi(optarg ? optarg : "");
            if (refresh_rate_hz <= 0) {
              throw std::runtime_error("-r/--refresh must be greater than 0.");
            }
            break;
          case 'h':
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                      << "Options:" << std::endl
                      << "  -h, --help     display this information" << std::endl
                      << "  --hololink     IP address of Hololink board (default `192.168.0.2`)"
                      << std::endl
                      << "  --camera-mode  Camera mode (default `"
                      << int(hololink::sensors::imx274_mode::IMX274_MODE_1920X1080_60FPS) << "`)"
                      << std::endl
                      << "  --gsync        Enable VRR (disabled by default)" << std::endl
                      << "  --fbr          Enable front-buffer rendering (enabled by default)"
                      << std::endl
                      << "  -r, --refresh  Display refresh rate in Hz (default `60`)" << std::endl
                      << "  -l, --resolution   Display resolution as WxH (default `2560x1440`)"
                      << std::endl
                      << std::endl;
            return EXIT_SUCCESS;

          default:
            throw std::runtime_error("Unhandled option ");
        }
      }
    }

    const bool front_buffer_rendering = !gsync;

    holoscan::set_log_level(log_level);

    std::cout << "Initializing." << std::endl;

    CudaCheck(cuInit(0));
    int cu_device_ordinal = 0;
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    CudaCheck(cuCtxSetCurrent(cu_context));

    // Get a handle to the data source
    hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
    hololink::DataChannel hololink_channel(channel_metadata);

    // Get a handle to the camera
    auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(hololink_channel,
                                                                          expander_configuration);
    camera->set_mode(camera_mode);

    // Calculate CSI frame size using the csi_to_bayer converter base
    auto csi_calculator = std::make_shared<hololink::operators::CsiToBayerConverterBase>();
    camera->configure_converter(csi_calculator);
    const size_t frame_size = csi_calculator->get_csi_length();

    const uint32_t width = camera->get_width();
    const uint32_t height = camera->get_height();
    const auto pixel_format = camera->get_pixel_format();
    const auto bayer_format = camera->get_bayer_format();

    auto shared_state = std::make_shared<SharedFrameState>();

    AppConfig config;
    config.cuda_context = cu_context;
    config.hololink_channel = &hololink_channel;
    config.ibv_name = ibv_name;
    config.ibv_port = ibv_port;
    config.camera = camera;
    config.camera_mode = camera_mode;
    config.frame_size = frame_size;
    config.width = width;
    config.height = height;
    config.pixel_format = pixel_format;
    config.bayer_format = bayer_format;
    config.gsync = gsync;
    config.front_buffer_rendering = front_buffer_rendering;
    config.refresh_rate_hz = refresh_rate_hz;
    config.display_width = display_width;
    config.display_height = display_height;
    config.shared_state = shared_state;

    auto application = holoscan::make_application<Imx274GrApplication>(config);
    application->config(configuration);

    // Compose the application graph to create the fragment objects
    application->compose_graph();

    // Get the GPU resident fragment and enable performance measurement
    auto& fragment_graph = application->fragment_graph();
    auto gr_fragment = fragment_graph.find_node("gr_fragment");
    if (gr_fragment) {
      // 100000 samples at max
      gr_fragment->gpu_resident().enable_perf_measurement(100000);
      gr_fragment->gpu_resident().data_not_ready_sleep_interval_us(100);
    }

    // Run it.
    std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
    hololink->start();
    hololink->reset();
    camera->setup_clock();
    camera->configure(camera_mode);
    camera->set_digital_gain_reg(0x4);
    if (pattern_set) {
      camera->test_pattern(pattern);
    }
    auto future = application->run_async();
    future.get();

    // Print and save performance metrics
    if (gr_fragment) {
      gr_fragment->gpu_resident().print_perf_metrics(100, 100);
      gr_fragment->gpu_resident().save_perf_results_as_csv();
    }

    hololink->stop();

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
