/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, Real-Time Innovations, Inc. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>

#include "dds_shapes_subscriber.hpp"
#include "dds_video_publisher.hpp"
#include "dds_video_subscriber.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <linux/videodev2.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

namespace holoscan::ops {

/**
 * @brief Operator to consume the shapes output from a DDSShapesSubscriber and
 * render them to Holoviz.
 */
class DDSShapesRenderer : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DDSShapesRenderer)

  DDSShapesRenderer() = default;

  void setup(OperatorSpec& spec) override {
    // Inputs from DDSShapesSubscriber
    spec.input<std::vector<holoscan::ops::DDSShapesSubscriberOp::Shape>>("input");

    // Outputs to Holoviz
    spec.output<gxf::Entity>("outputs");
    spec.output<std::vector<HolovizOp::InputSpec>>("output_specs");

    spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  }

  void initialize() override {
    Operator::initialize();
  }

  /**
   * @brief Helper function to add a tensor with data to an entity.
   */
  template <std::size_t N, std::size_t C>
  void add_data(gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context) {
    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    // Add a tensor
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
    // Reshape the tensor to the size of the data
    tensor->reshape<float>(
        nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    // Copy the data to the tensor
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto specs = std::vector<HolovizOp::InputSpec>();

    // Get the input shapes from the DDSShapesSubscriber.
    auto shapes = op_input.receive<std::vector<
        holoscan::ops::DDSShapesSubscriberOp::Shape>>("input").value();

    // Generate the output Holoviz specs and primitives.
    int32_t shape_id = 0;
    for (const auto& shape : shapes) {
      const auto shape_name = std::to_string(shape_id);
      auto& spec = specs.emplace_back();
      spec.tensor_name_ = shape_name;
      spec.color_ = shape.color_;
      spec.priority_ = shape_id++;
      spec.line_width_ = shape_line_width_;
      if (shape.type_ == DDSShapesSubscriberOp::Shape::Type::SQUARE) {
        spec.type_ = HolovizOp::InputType::RECTANGLES;
        add_data<2, 2>(entity, shape_name.c_str(),
            {{{shape.x_ - shape.width_ / 2, shape.y_ - shape.height_ / 2},
              {shape.x_ + shape.width_ / 2, shape.y_ + shape.height_ / 2}}}, context);
      } else if (shape.type_ == DDSShapesSubscriberOp::Shape::Type::CIRCLE) {
        spec.type_ = HolovizOp::InputType::OVALS;
        add_data<1, 4>(entity, shape_name.c_str(),
            {{{shape.x_, shape.y_, shape.width_, shape.height_}}}, context);
      } else if (shape.type_ == DDSShapesSubscriberOp::Shape::Type::TRIANGLE) {
        spec.type_ = HolovizOp::InputType::LINE_STRIP;
        add_data<4, 2>(entity, shape_name.c_str(),
            {{{shape.x_ - shape.width_ / 2, shape.y_ + shape.height_ / 2},
              {shape.x_ + shape.width_ / 2, shape.y_ + shape.height_ / 2},
              {shape.x_,                    shape.y_ - shape.height_ / 2},
              {shape.x_ - shape.width_ / 2, shape.y_ + shape.height_ / 2}}}, context);
      }
    }

    // Output to Holoviz.
    op_output.emit(entity, "outputs");
    op_output.emit(specs, "output_specs");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;

  const float shape_line_width_ = 5.0f;
};

/** Produces an animated RGBA test pattern using RTI white, blue, and orange. */
class SyntheticVideoSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SyntheticVideoSourceOp)

  void setup(OperatorSpec& spec) override {
    spec.output<gxf::Entity>("signal");
    spec.param(allocator_, "allocator", "Allocator", "Allocator for synthetic video frames");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_->gxf_cid());
    auto output = nvidia::gxf::Entity::New(context.context());
    if (!output) { throw std::runtime_error("Failed to allocate synthetic video entity"); }

    auto video_buffer = output.value().add<nvidia::gxf::VideoBuffer>();
    if (!video_buffer) { throw std::runtime_error("Failed to allocate synthetic video buffer"); }
    video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
        width_, height_, nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
        nvidia::gxf::MemoryStorageType::kHost, allocator.value());

    auto* pixels = static_cast<uint8_t*>(video_buffer.value()->pointer());
    const auto& info = video_buffer.value()->video_frame_info();
    const size_t stride = info.color_planes[0].stride;
    const uint32_t marker_x = (frame_number_ * 8) % width_;
    for (uint32_t y = 0; y < height_; ++y) {
      for (uint32_t x = 0; x < width_; ++x) {
        uint8_t r = 255, g = 255, b = 255;
        if (y >= height_ / 3 && y < 2 * height_ / 3) {
          r = 0;
          g = 77;
          b = 151;  // RTI blue
        } else if (y >= 2 * height_ / 3) {
          r = 247;
          g = 148;
          b = 30;  // RTI orange
        }
        if (x >= marker_x && x < std::min(marker_x + 12, width_)) {
          r = 20;
          g = 20;
          b = 20;
        }
        auto* pixel = pixels + y * stride + x * 4;
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
        pixel[3] = 255;
      }
    }
    ++frame_number_;
    auto entity = gxf::Entity(std::move(output.value()));
    op_output.emit(entity, "signal");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  static constexpr uint32_t width_ = 640;
  static constexpr uint32_t height_ = 480;
  uint32_t frame_number_ = 0;
};

/** Validates DDS video frames when no graphical display is available. */
class HeadlessVideoSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HeadlessVideoSinkOp)
  HeadlessVideoSinkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("input");
    spec.param(screenshot_path_, "screenshot_path", "Screenshot path",
               "Write the first received DDS frame as a PPM image", std::string());
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto entity = op_input.receive<gxf::Entity>("input").value();
    const auto buffer =
        static_cast<nvidia::gxf::Entity&>(entity).get<nvidia::gxf::VideoBuffer>();
    if (!buffer) { throw std::runtime_error("DDS sample does not contain a VideoBuffer"); }
    const auto& info = buffer.value()->video_frame_info();
    ++frame_count_;
    if (!screenshot_written_ && !screenshot_path_.get().empty()) {
      std::ofstream screenshot(screenshot_path_.get(), std::ios::binary);
      if (!screenshot) {
        throw std::runtime_error("Failed to open DDS video screenshot output");
      }
      screenshot << "P6\n" << info.width << ' ' << info.height << "\n255\n";
      const auto* pixels = static_cast<const uint8_t*>(buffer.value()->pointer());
      const size_t stride = info.color_planes[0].stride;
      for (uint32_t y = 0; y < info.height; ++y) {
        for (uint32_t x = 0; x < info.width; ++x) {
          const auto* pixel = pixels + y * stride + x * 4;
          screenshot.write(reinterpret_cast<const char*>(pixel), 3);
        }
      }
      screenshot.close();
      if (!screenshot) { throw std::runtime_error("Failed to write DDS video screenshot"); }
      screenshot_written_ = true;
      HOLOSCAN_LOG_INFO("Wrote first received DDS video frame to {}", screenshot_path_.get());
    }
    if (frame_count_ == 1 || frame_count_ % 30 == 0) {
      HOLOSCAN_LOG_INFO("Received {} DDS video frames ({}x{})",
                        frame_count_, info.width, info.height);
    }
  }

 private:
  Parameter<std::string> screenshot_path_;
  uint64_t frame_count_ = 0;
  bool screenshot_written_ = false;
};

}  // namespace holoscan::ops

namespace {

constexpr uint32_t kVideoWidth = 640;
constexpr uint32_t kVideoHeight = 480;

bool camera_supports_video_mode(const char* device, uint32_t width, uint32_t height) {
  const int video_fd = open(device, O_RDWR | O_NONBLOCK);
  if (video_fd < 0) { return false; }

  struct v4l2_capability video_caps {};
  if (ioctl(video_fd, VIDIOC_QUERYCAP, &video_caps) != 0) {
    close(video_fd);
    return false;
  }

  const uint32_t capabilities =
      (video_caps.capabilities & V4L2_CAP_DEVICE_CAPS)
          ? video_caps.device_caps
          : video_caps.capabilities;
  if (!(capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    close(video_fd);
    return false;
  }

  bool mode_supported = false;
  for (uint32_t index = 0; ; ++index) {
    struct v4l2_fmtdesc format_description {};
    format_description.index = index;
    format_description.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(video_fd, VIDIOC_ENUM_FMT, &format_description) != 0) { break; }

    struct v4l2_format format {};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = width;
    format.fmt.pix.height = height;
    format.fmt.pix.pixelformat = format_description.pixelformat;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(video_fd, VIDIOC_TRY_FMT, &format) == 0 &&
        format.fmt.pix.width == width && format.fmt.pix.height == height) {
      mode_supported = true;
      break;
    }
  }

  close(video_fd);
  return mode_supported;
}

}  // namespace


/**
 * @brief Application to publish a V4L2 video stream to DDS.
 */
class VideoToDDS : public holoscan::Application {
 public:
  explicit VideoToDDS(uint32_t domain_id, uint32_t stream_id, bool synthetic,
                      uint32_t camera_width, uint32_t camera_height)
      : domain_id_(domain_id),
        stream_id_(stream_id),
        synthetic_(synthetic),
        camera_width_(camera_width),
        camera_height_(camera_height) {}

  void compose() override {
    using namespace holoscan;

    auto allocator = make_resource<UnboundedAllocator>("pool");
    std::shared_ptr<Operator> source;
    if (synthetic_) {
      source = make_operator<ops::SyntheticVideoSourceOp>(
          "synthetic_source",
          make_condition<PeriodicCondition>("synthetic_frame_rate", std::chrono::milliseconds(33)),
          Arg("allocator", allocator));
    } else {
      source = make_operator<ops::V4L2VideoCaptureOp>("v4l2",
          Arg("allocator", allocator),
          Arg("width", camera_width_),
          Arg("height", camera_height_),
          Arg("pixel_format", std::string("auto")));
    }

    auto dds = make_operator<ops::DDSVideoPublisherOp>("dds",
        Arg("participant_qos", std::string("HoloscanDDSTransport::SHMEM+LAN")),
        Arg("writer_qos", std::string("HoloscanDDSDataFlow::Video")),
        Arg("domain_id", domain_id_),
        Arg("stream_id", stream_id_));

    add_flow(source, dds, {{"signal", "input"}});
  }

 private:
  uint32_t domain_id_;
  uint32_t stream_id_;
  bool synthetic_;
  uint32_t camera_width_;
  uint32_t camera_height_;
};

/**
 * @brief Application to render a DDS video stream (published by the DDSVideoPublisher)
 * and shapes (published by the RTI Connext Shapes Demo) to Holoviz.
 */
class DDSToHoloviz : public holoscan::Application {
 public:
  explicit DDSToHoloviz(uint32_t domain_id, uint32_t stream_id, bool headless,
                        std::string screenshot_path)
      : domain_id_(domain_id),
        stream_id_(stream_id),
        headless_(headless),
        screenshot_path_(std::move(screenshot_path)) {}

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<UnboundedAllocator> allocator = make_resource<UnboundedAllocator>("pool");

    //  DDS Video Subscriber
    auto participant_qos = std::string("HoloscanDDSTransport::SHMEM+LAN");
    auto video_subscriber = make_operator<ops::DDSVideoSubscriberOp>("video_subscriber",
        Arg("allocator", allocator),
        Arg("domain_id", domain_id_),
        Arg("stream_id", stream_id_),
        Arg("participant_qos", participant_qos),
        Arg("reader_qos", std::string("HoloscanDDSDataFlow::Video")));

    if (headless_) {
      auto sink = make_operator<ops::HeadlessVideoSinkOp>(
          "headless_video_sink", Arg("screenshot_path", screenshot_path_));
      add_flow(video_subscriber, sink, {{"output", "input"}});
      return;
    }

    // DDS Shapes Subscriber
    auto shapes_subscriber = make_operator<ops::DDSShapesSubscriberOp>("shapes_subscriber",
        Arg("domain_id", domain_id_),
        Arg("participant_qos", participant_qos),
        Arg("reader_qos", std::string("HoloscanDDSDataFlow::Shapes")));

    // DDS Shapes Renderer
    auto shapes_renderer = make_operator<ops::DDSShapesRenderer>("shapes_renderer",
        Arg("allocator", allocator));

    // Holoviz (initialize with the default input spec for the video stream)
    std::vector<ops::HolovizOp::InputSpec> input_spec;
    auto& video_spec = input_spec.emplace_back(
        ops::HolovizOp::InputSpec("", ops::HolovizOp::InputType::COLOR));
    auto holoviz = make_operator<ops::HolovizOp>("holoviz",
        Arg("width", kVideoWidth), Arg("height", kVideoHeight), Arg("tensors", input_spec),
        Arg("headless", headless_));

    add_flow(video_subscriber, holoviz, {{"output", "receivers"}});
    add_flow(shapes_subscriber, shapes_renderer, {{"output", "input"}});
    add_flow(shapes_renderer, holoviz, {{"outputs", "receivers"}, {"output_specs", "input_specs"}});
  }

 private:
  uint32_t domain_id_;
  uint32_t stream_id_;
  bool headless_;
  std::string screenshot_path_;
};

void usage() {
  std::cout << "Usage: dds_video {-p | -s} [options]" << std::endl << std::endl
            << "Options" << std::endl
            << "  -p,    --publisher    Run as a publisher" << std::endl
            << "  -s,    --subscriber   Run as a subscriber" << std::endl
            << "  -t,    --synthetic    Publish an animated RTI test pattern instead of V4L2"
            << std::endl
            << "  -n,    --no-display   Validate received frames without opening a display"
            << std::endl
            << "  -o P,  --screenshot=P Write the first received frame as a PPM image"
            << std::endl
            << "  -d ID, --domain=ID    Use the specified DDS domain ID" << std::endl
            << "  -i ID, --id=ID        Use the specified video stream ID" << std::endl;
}

int main(int argc, char** argv) {
  bool publisher = false;
  bool subscriber = false;
  bool synthetic = false;
  bool no_display = false;
  std::string screenshot_path;
  uint32_t stream_id = 0;
  uint32_t domain_id = 0;

  struct option long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"publisher", no_argument, 0, 'p'},
      {"subscriber", no_argument, 0, 's'},
      {"synthetic", no_argument, 0, 't'},
      {"no-display", no_argument, 0, 'n'},
      {"screenshot", required_argument, 0, 'o'},
      {"id", required_argument, 0, 'i'},
      {"domain", required_argument, 0, 'd'},
      {0, 0, 0, 0}};

  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hpstno:i:d:", long_options, &option_index);
    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        usage();
        return 0;
      case 'p':
        publisher = true;
        break;
      case 's':
        subscriber = true;
        break;
      case 't':
        synthetic = true;
        break;
      case 'n':
        no_display = true;
        break;
      case 'o':
        screenshot_path = argument;
        break;
      case 'i':
        stream_id = stoi(argument);
        break;
      case 'd':
        domain_id = stoi(argument);
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unhandled option '{}'", static_cast<char>(c));
    }
  }

  if (publisher == subscriber) {
    HOLOSCAN_LOG_ERROR("Must provide either -p or -s for publisher or subscriber, respectively");
    usage();
    return -1;
  }
  if (synthetic && !publisher) {
    HOLOSCAN_LOG_ERROR("--synthetic is only valid with --publisher");
    return 2;
  }
  if (no_display && !subscriber) {
    HOLOSCAN_LOG_ERROR("--no-display is only valid with --subscriber");
    return 2;
  }
  if (!screenshot_path.empty() && !subscriber) {
    HOLOSCAN_LOG_ERROR("--screenshot is only valid with --subscriber");
    return 2;
  }

  HOLOSCAN_LOG_INFO("Starting {} for stream {} in domain {}",
      publisher ? "publisher" : "subscriber", stream_id, domain_id);

  if (publisher) {
    uint32_t camera_width = kVideoWidth;
    uint32_t camera_height = kVideoHeight;
    if (!synthetic) {
      constexpr std::array<std::array<uint32_t, 2>, 3> preferred_modes = {{
          {640, 480},
          {1280, 720},
          {1920, 1080},
      }};
      bool camera_available = false;
      for (const auto& mode : preferred_modes) {
        if (camera_supports_video_mode("/dev/video0", mode[0], mode[1])) {
          camera_width = mode[0];
          camera_height = mode[1];
          camera_available = true;
          break;
        }
      }
      if (!camera_available) {
        HOLOSCAN_LOG_ERROR(
            "Publisher requires a compatible V4L2 camera at /dev/video0; "
            "use --synthetic when no compatible camera is available");
        return 2;
      }
      HOLOSCAN_LOG_INFO("Using V4L2 camera at {}x{}", camera_width, camera_height);
    }
    auto app = holoscan::make_application<VideoToDDS>(
        domain_id, stream_id, synthetic, camera_width, camera_height);
    try {
      app->run();
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_ERROR("DDS video publisher stopped: {}", error.what());
      return 2;
    }
  } else if (subscriber) {
    // Holoviz must run headless when no display is available (for example in
    // CI or an SSH session). This avoids initializing a GUI backend that can
    // terminate the process before DDS discovery begins.
    const bool headless = no_display || !screenshot_path.empty() ||
                          (std::getenv("DISPLAY") == nullptr &&
                           std::getenv("WAYLAND_DISPLAY") == nullptr);
    auto app = holoscan::make_application<DDSToHoloviz>(
        domain_id, stream_id, headless, std::move(screenshot_path));
    try {
      app->run();
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_ERROR("DDS video subscriber stopped: {}", error.what());
      return 2;
    }
  }

  return 0;
}
