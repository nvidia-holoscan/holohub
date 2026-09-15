/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, TECNALIA. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gxf/multimedia/video.hpp>
#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

#include "nv_video_decoder.hpp"

namespace holoscan::ops::nv_video_decoder_test {

constexpr std::array<std::size_t, 3> kAccessUnitSizes = {635, 383, 507};
constexpr std::array<uint8_t, 3> kExpectedLumaValues = {16, 96, 200};
constexpr uint32_t kFrameWidth = 256;
constexpr uint32_t kFrameHeight = 256;
constexpr int kPacketBatchId = 73;

// data/three_access_units.h265 is a frozen Annex-B HEVC fixture generated from three
// lossless 256x256 YUV420p frames with Y={16,96,200}, U=V=128 using libx265 with:
// lossless=1:bframes=0:keyint=30:min-keyint=30:scenecut=0:repeat-headers=1:
// aud=0:annexb=1:info=0. ffprobe reports AU sizes {635,383,507} bytes.
// SHA-256: 0ed2b3e4ef695e82f19a7c7cf450bd924204c509a2c92f18ecac03896c4c56b9

std::vector<uint8_t> read_fixture(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Failed to open HEVC fixture: " + path);
  }

  const auto end_position = input.tellg();
  if (end_position < 0) {
    throw std::runtime_error("Failed to determine HEVC fixture size: " + path);
  }

  const auto file_size = static_cast<std::size_t>(end_position);
  std::vector<uint8_t> data(file_size);
  input.seekg(0, std::ios::beg);
  if (!input.read(reinterpret_cast<char*>(data.data()),
                  static_cast<std::streamsize>(file_size))) {
    throw std::runtime_error("Failed to read HEVC fixture: " + path);
  }
  return data;
}

class AccessUnitSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AccessUnitSourceOp)
  AccessUnitSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_,
               "allocator",
               "Allocator",
               "Allocator for encoded access-unit tensors.",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("source_allocator")));
    spec.param(fixture_path_,
               "fixture_path",
               "FixturePath",
               "Path to the three-access-unit HEVC fixture.");
    spec.param(combine_access_units_,
               "combine_access_units",
               "CombineAccessUnits",
               "Emit all access units in one tensor.",
               false);
    spec.param(signal_end_of_stream_,
               "signal_end_of_stream",
               "SignalEndOfStream",
               "Mark the final tensor as the end of the packetized stream.",
               false);
    spec.param(wide_elements_,
               "wide_elements",
               "WideElements",
               "Store encoded bytes in a uint16 tensor.",
               false);
    spec.output<nvidia::gxf::Entity>("output");
  }

  void initialize() override {
    add_arg(allocator_.default_value());
    Operator::initialize();

    bitstream_ = read_fixture(fixture_path_.get());
    const std::size_t expected_size =
        std::accumulate(kAccessUnitSizes.begin(), kAccessUnitSizes.end(), std::size_t{0});
    if (bitstream_.size() != expected_size) {
      throw std::runtime_error("Unexpected HEVC fixture size");
    }
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    const std::size_t access_unit_size = combine_access_units_.get()
                                             ? bitstream_.size()
                                             : kAccessUnitSizes.at(next_access_unit_++);
    auto entity = gxf::Entity::New(&context);
    auto tensor =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>().value();
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());
    if (wide_elements_.get()) {
      const auto element_count = (access_unit_size + sizeof(uint16_t) - 1) / sizeof(uint16_t);
      tensor->reshape<uint16_t>(nvidia::gxf::Shape({static_cast<int32_t>(element_count)}),
                                nvidia::gxf::MemoryStorageType::kHost,
                                gxf_allocator.value());
      std::memset(tensor->pointer(), 0, tensor->nbytes());
    } else {
      tensor->reshape<uint8_t>(nvidia::gxf::Shape({static_cast<int32_t>(access_unit_size)}),
                               nvidia::gxf::MemoryStorageType::kHost,
                               gxf_allocator.value());
    }
    std::memcpy(tensor->pointer(), bitstream_.data() + next_offset_, access_unit_size);
    next_offset_ += access_unit_size;
    if (signal_end_of_stream_.get()) {
      metadata()->set("end_of_stream", next_offset_ == bitstream_.size());
    }
    metadata()->set("packet_batch_id", kPacketBatchId);

    op_output.emit(entity, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> fixture_path_;
  Parameter<bool> combine_access_units_;
  Parameter<bool> signal_end_of_stream_;
  Parameter<bool> wide_elements_;
  std::vector<uint8_t> bitstream_;
  std::size_t next_access_unit_ = 0;
  std::size_t next_offset_ = 0;
};

class FrameValidationSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameValidationSinkOp)
  FrameValidationSinkOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("input"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<gxf::Entity>("input");
    if (!maybe_entity) {
      throw std::runtime_error("Failed to receive decoded frame");
    }
    if (metadata()->get<int>("packet_batch_id", -1) != kPacketBatchId) {
      throw std::runtime_error("Decoded frame did not preserve its input metadata");
    }

    auto maybe_video_buffer = static_cast<nvidia::gxf::Entity&>(maybe_entity.value())
                                  .get<nvidia::gxf::VideoBuffer>();
    if (!maybe_video_buffer) {
      throw std::runtime_error("Decoded entity does not contain a video buffer");
    }

    auto video_buffer = maybe_video_buffer.value();
    if (video_buffer->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
      throw std::runtime_error("Decoded video buffer is not in device memory");
    }
    const auto& info = video_buffer->video_frame_info();
    if (info.width != kFrameWidth || info.height != kFrameHeight) {
      throw std::runtime_error("Unexpected decoded frame dimensions: " +
                               std::to_string(info.width) + "x" +
                               std::to_string(info.height));
    }
    if (info.color_format != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12 ||
        info.color_planes.empty()) {
      throw std::runtime_error("Decoded frame is not NV12");
    }
    if (frame_count_ >= kExpectedLumaValues.size()) {
      throw std::runtime_error("Received more decoded frames than the fixture contains");
    }

    const auto& luma_plane = info.color_planes[0];
    std::vector<uint8_t> luma(kFrameWidth * kFrameHeight);
    const auto copy_result =
        cudaMemcpy2D(luma.data(),
                     kFrameWidth,
                     video_buffer->pointer() + luma_plane.offset,
                     luma_plane.stride,
                     kFrameWidth,
                     kFrameHeight,
                     cudaMemcpyDeviceToHost);
    if (copy_result != cudaSuccess) {
      throw std::runtime_error("Failed to copy decoded luma plane to host: " +
                               std::string(cudaGetErrorString(copy_result)));
    }

    const uint8_t expected_luma = kExpectedLumaValues[frame_count_];
    const auto mismatch =
        std::find_if(luma.begin(), luma.end(), [expected_luma](uint8_t value) {
          return value != expected_luma;
        });
    if (mismatch != luma.end()) {
      throw std::runtime_error("Unexpected luma value in decoded frame " +
                               std::to_string(frame_count_) + ": expected " +
                               std::to_string(expected_luma) + ", got " +
                               std::to_string(*mismatch));
    }

    ++frame_count_;
  }

  std::size_t frame_count() const { return frame_count_; }

 private:
  std::size_t frame_count_ = 0;
};

class PacketizedFramingApp : public Application {
 public:
  PacketizedFramingApp(std::string fixture_path, std::string input_mode, bool combine_access_units,
                       bool signal_end_of_stream, bool wide_elements)
      : fixture_path_(std::move(fixture_path)),
        input_mode_(std::move(input_mode)),
        combine_access_units_(combine_access_units),
        signal_end_of_stream_(signal_end_of_stream),
        wide_elements_(wide_elements) {}

  void compose() override {
    auto source = make_operator<AccessUnitSourceOp>(
        "source",
        Arg("fixture_path", fixture_path_),
        Arg("combine_access_units", combine_access_units_),
        Arg("signal_end_of_stream", signal_end_of_stream_),
        Arg("wide_elements", wide_elements_),
        make_condition<CountCondition>(
            "source_count", combine_access_units_ ? 1 : kAccessUnitSizes.size()));
    auto decoder = make_operator<NvVideoDecoderOp>(
        "decoder",
        Arg("cuda_device_ordinal", 0),
        Arg("allocator", make_resource<UnboundedAllocator>("decoder_allocator")),
        Arg("codec", std::string("HEVC")),
        Arg("packetized_input_mode", input_mode_),
        // Preserve the no-EOS framing tests in low-latency mode. The EOS regression
        // instead exercises the default display/reordering policy and its final drain.
        Arg("packetized_low_latency", !signal_end_of_stream_));
    sink_ = make_operator<FrameValidationSinkOp>("sink");

    // Keep the default connector capacity of one. Combined-input tests verify
    // that multi-frame decoder returns are queued and emitted across ticks.
    add_flow(source, decoder, {{"output", "input"}});
    add_flow(decoder, sink_, {{"output", "input"}});
  }

  std::shared_ptr<FrameValidationSinkOp> sink_;

 private:
  std::string fixture_path_;
  std::string input_mode_;
  bool combine_access_units_;
  bool signal_end_of_stream_;
  bool wide_elements_;
};

}  // namespace holoscan::ops::nv_video_decoder_test

int main(int argc, char** argv) {
  using holoscan::ops::nv_video_decoder_test::PacketizedFramingApp;

  if (argc != 7) {
    std::cerr << "Usage: nv_video_decoder_packetized_framing_test "
                 "<fixture.h265> <stream|access_unit> <expected_frames> "
                 "<separate|combined> <no_eos|eos> <uint8|uint16>\n";
    return 2;
  }

  const std::string input_mode = argv[2];
  if (input_mode != "stream" && input_mode != "access_unit") {
    std::cerr << "Unsupported packetized input mode: " << input_mode << '\n';
    return 2;
  }

  const std::string tensor_mode = argv[4];
  if (tensor_mode != "separate" && tensor_mode != "combined") {
    std::cerr << "Unsupported tensor mode: " << tensor_mode << '\n';
    return 2;
  }

  const std::string eos_mode = argv[5];
  if (eos_mode != "no_eos" && eos_mode != "eos") {
    std::cerr << "Unsupported EOS mode: " << eos_mode << '\n';
    return 2;
  }

  const std::string element_type = argv[6];
  if (element_type != "uint8" && element_type != "uint16") {
    std::cerr << "Unsupported tensor element type: " << element_type << '\n';
    return 2;
  }

  try {
    const std::size_t expected_frames = std::stoul(argv[3]);
    auto app = holoscan::make_application<PacketizedFramingApp>(
        argv[1],
        input_mode,
        tensor_mode == "combined",
        eos_mode == "eos",
        element_type == "uint16");
    app->run();

    const std::size_t actual_frames = app->sink_->frame_count();
    std::cout << "packetized_input_mode=" << input_mode << ": expected " << expected_frames
              << " decoded frames, got " << actual_frames << '\n';
    if (actual_frames != expected_frames) {
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "NvVideoDecoderOp packetized framing test failed: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
