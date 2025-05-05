/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <getopt.h>

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

}  // namespace holoscan::ops


/**
 * @brief Application to publish a V4L2 video stream to DDS.
 */
class V4L2ToDDS : public holoscan::Application {
 public:
  explicit V4L2ToDDS(uint32_t domain_id, uint32_t stream_id)
      : domain_id_(domain_id), stream_id_(stream_id) {}

  void compose() override {
    using namespace holoscan;

    auto v4l2 = make_operator<ops::V4L2VideoCaptureOp>("v4l2",
        Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
        Arg("width", 640u), Arg("height", 480u));

    auto dds = make_operator<ops::DDSVideoPublisherOp>("dds",
        Arg("participant_qos", std::string("HoloscanDDSTransport::SHMEM+LAN")),
        Arg("writer_qos", std::string("HoloscanDDSDataFlow::Video")),
        Arg("domain_id", domain_id_),
        Arg("stream_id", stream_id_));

    add_flow(v4l2, dds, {{"signal", "input"}});
  }

 private:
  uint32_t domain_id_;
  uint32_t stream_id_;
};

/**
 * @brief Application to render a DDS video stream (published by the DDSVideoPublisher)
 * and shapes (published by the RTI Connext Shapes Demo) to Holoviz.
 */
class DDSToHoloviz : public holoscan::Application {
 public:
  explicit DDSToHoloviz(uint32_t domain_id, uint32_t stream_id)
      : domain_id_(domain_id), stream_id_(stream_id) {}

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
        Arg("width", 640u), Arg("height", 480u), Arg("tensors", input_spec));

    add_flow(video_subscriber, holoviz, {{"output", "receivers"}});
    add_flow(shapes_subscriber, shapes_renderer, {{"output", "input"}});
    add_flow(shapes_renderer, holoviz, {{"outputs", "receivers"}, {"output_specs", "input_specs"}});
  }

 private:
  uint32_t domain_id_;
  uint32_t stream_id_;
};

void usage() {
  std::cout << "Usage: dds_video {-p | -s} [options]" << std::endl << std::endl
            << "Options" << std::endl
            << "  -p,    --publisher    Run as a publisher" << std::endl
            << "  -s,    --subscriber   Run as a subscriber" << std::endl
            << "  -d ID, --domain=ID    Use the specified DDS domain ID" << std::endl
            << "  -i ID, --id=ID        Use the specified video stream ID" << std::endl;
}

int main(int argc, char** argv) {
  bool publisher = false;
  bool subscriber = false;
  uint32_t stream_id = 0;
  uint32_t domain_id = 0;

  struct option long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"publisher", no_argument, 0, 'p'},
      {"subscriber", no_argument, 0, 's'},
      {"id", required_argument, 0, 'i'},
      {"domain", required_argument, 0, 'd'},
      {0, 0, 0, 0}};

  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hpsi:d:", long_options, &option_index);
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

  if (!publisher && !subscriber) {
    HOLOSCAN_LOG_ERROR("Must provide either -p or -s for publisher or subscriber, respectively");
    usage();
    return -1;
  }

  HOLOSCAN_LOG_INFO("Starting {} for stream {} in domain {}",
      publisher ? "publisher" : "subscriber", stream_id, domain_id);

  if (publisher) {
    auto app = holoscan::make_application<V4L2ToDDS>(domain_id, stream_id);
    app->run();
  } else if (subscriber) {
    auto app = holoscan::make_application<DDSToHoloviz>(domain_id, stream_id);
    app->run();
  }

  return 0;
}
