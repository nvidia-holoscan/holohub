/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, TECNALIA. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gxf/multimedia/video.hpp>
#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

#include "nv_video_decoder.hpp"

namespace holoscan::ops::nv_video_decoder_test {

constexpr char kDisplayAreaChangeError[] =
    "Packetized display-area change requires stream_reset";

// data/crop250.h264 and data/crop256.h264 are the reviewer's two-frame H.264
// fixtures, generated from 256x250 black and 256x256 white color sources with
// libx264 and bframes=0:repeat-headers=1. Their SHA-256 sums are, respectively,
// e188fa9cbfdfa5db2f96008f3864e3f9a0999ffaffafff5f0c351a9d01782ed6 and
// f2c5254d97802278fab3d87de2031d6bce83046b7ccd65c53fb072c32e11f3ea.
std::vector<uint8_t> read_fixture(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Failed to open decoder fixture: " + path);
  }

  const auto end_position = input.tellg();
  if (end_position < 0) {
    throw std::runtime_error("Failed to determine decoder fixture size: " + path);
  }

  std::vector<uint8_t> data(static_cast<std::size_t>(end_position));
  input.seekg(0, std::ios::beg);
  if (!input.read(reinterpret_cast<char*>(data.data()),
                  static_cast<std::streamsize>(data.size()))) {
    throw std::runtime_error("Failed to read decoder fixture: " + path);
  }
  return data;
}

class DisplayCropSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DisplayCropSourceOp)
  DisplayCropSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_,
               "allocator",
               "Allocator",
               "Allocator for encoded tensors.",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("source_allocator")));
    spec.param(crop250_path_, "crop250_path", "Crop250Path", "Path to the 256x250 clip.");
    spec.param(crop256_path_, "crop256_path", "Crop256Path", "Path to the 256x256 clip.");
    spec.param(mode_, "mode", "Mode", "Input sequence: crop250, crop256, reject, or reset.");
    spec.output<nvidia::gxf::Entity>("output");
  }

  void initialize() override {
    add_arg(allocator_.default_value());
    Operator::initialize();
    crop250_ = read_fixture(crop250_path_.get());
    crop256_ = read_fixture(crop256_path_.get());
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    const auto& mode = mode_.get();
    std::vector<uint8_t> combined;
    const std::vector<uint8_t>* payload = nullptr;
    int64_t expected_height = 0;

    if (mode == "crop250") {
      payload = &crop250_;
      expected_height = 250;
    } else if (mode == "crop256") {
      payload = &crop256_;
      expected_height = 256;
    } else if (mode == "reject") {
      combined.reserve(crop250_.size() + crop256_.size());
      combined.insert(combined.end(), crop250_.begin(), crop250_.end());
      combined.insert(combined.end(), crop256_.begin(), crop256_.end());
      payload = &combined;
    } else if (mode == "reset") {
      payload = emission_index_ == 0 ? &crop250_ : &crop256_;
      expected_height = emission_index_ == 0 ? 250 : 256;
      metadata()->set("stream_reset", emission_index_ != 0);
    } else {
      throw std::runtime_error("Unsupported display-crop test mode: " + mode);
    }

    auto entity = gxf::Entity::New(&context);
    auto tensor =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>().value();
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());
    tensor->reshape<uint8_t>(nvidia::gxf::Shape({static_cast<int32_t>(payload->size())}),
                             nvidia::gxf::MemoryStorageType::kHost,
                             gxf_allocator.value());
    std::memcpy(tensor->pointer(), payload->data(), payload->size());

    metadata()->set("expected_height", expected_height);
    metadata()->set("end_of_stream", true);
    ++emission_index_;
    op_output.emit(entity, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> crop250_path_;
  Parameter<std::string> crop256_path_;
  Parameter<std::string> mode_;
  std::vector<uint8_t> crop250_;
  std::vector<uint8_t> crop256_;
  std::size_t emission_index_ = 0;
};

class DisplayCropSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DisplayCropSinkOp)
  DisplayCropSinkOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("input"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<gxf::Entity>("input");
    if (!maybe_entity) {
      throw std::runtime_error("Failed to receive decoded frame");
    }
    auto maybe_video_buffer = static_cast<nvidia::gxf::Entity&>(maybe_entity.value())
                                  .get<nvidia::gxf::VideoBuffer>();
    if (!maybe_video_buffer) {
      throw std::runtime_error("Decoded entity does not contain a video buffer");
    }

    const auto& info = maybe_video_buffer.value()->video_frame_info();
    const auto expected_height = metadata()->get<int64_t>("expected_height", 0);
    if (info.width != 256 || info.height != expected_height) {
      throw std::runtime_error("Unexpected decoded frame dimensions: " +
                               std::to_string(info.width) + "x" +
                               std::to_string(info.height));
    }
    ++frame_counts_.at(expected_height == 250 ? 0 : 1);
  }

  std::size_t frame_count(int height) const { return frame_counts_.at(height == 250 ? 0 : 1); }

 private:
  std::array<std::size_t, 2> frame_counts_{};
};

class DisplayCropApp : public Application {
 public:
  DisplayCropApp(std::string crop250_path, std::string crop256_path, std::string mode)
      : crop250_path_(std::move(crop250_path)),
        crop256_path_(std::move(crop256_path)),
        mode_(std::move(mode)) {}

  void compose() override {
    auto source = make_operator<DisplayCropSourceOp>(
        "source",
        Arg("crop250_path", crop250_path_),
        Arg("crop256_path", crop256_path_),
        Arg("mode", mode_),
        make_condition<CountCondition>("source_count", mode_ == "reset" ? 2 : 1));
    auto decoder = make_operator<NvVideoDecoderOp>(
        "decoder",
        Arg("cuda_device_ordinal", 0),
        Arg("allocator", make_resource<UnboundedAllocator>("decoder_allocator")),
        Arg("codec", std::string("H264")),
        Arg("packetized_input_mode", std::string("stream")),
        Arg("packetized_low_latency", false));
    sink_ = make_operator<DisplayCropSinkOp>("sink");
    add_flow(source, decoder, {{"output", "input"}});
    add_flow(decoder, sink_, {{"output", "input"}});
  }

  std::shared_ptr<DisplayCropSinkOp> sink_;

 private:
  std::string crop250_path_;
  std::string crop256_path_;
  std::string mode_;
};

}  // namespace holoscan::ops::nv_video_decoder_test

int main(int argc, char** argv) {
  using holoscan::ops::nv_video_decoder_test::DisplayCropApp;
  using holoscan::ops::nv_video_decoder_test::kDisplayAreaChangeError;

  if (argc != 4) {
    std::cerr << "Usage: nv_video_decoder_display_crop_test "
                 "<crop250.h264> <crop256.h264> <crop250|crop256|reject|reset>\n";
    return 2;
  }

  const std::string mode = argv[3];
  try {
    auto app = holoscan::make_application<DisplayCropApp>(argv[1], argv[2], mode);
    app->run();

    if (mode == "reject") {
      std::cerr << "Expected the display-area change to require stream_reset\n";
      return 1;
    }

    const auto crop250_frames = app->sink_->frame_count(250);
    const auto crop256_frames = app->sink_->frame_count(256);
    if ((mode == "crop250" && (crop250_frames != 2 || crop256_frames != 0)) ||
        (mode == "crop256" && (crop250_frames != 0 || crop256_frames != 2)) ||
        (mode == "reset" && (crop250_frames != 2 || crop256_frames != 2))) {
      std::cerr << "Unexpected decoded frame counts: crop250=" << crop250_frames
                << ", crop256=" << crop256_frames << '\n';
      return 1;
    }
  } catch (const std::exception& e) {
    if (mode == "reject" && std::string(e.what()).find(kDisplayAreaChangeError) !=
                                std::string::npos) {
      std::cout << e.what() << '\n';
      return 0;
    }
    std::cerr << "NvVideoDecoderOp display-crop test failed: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
