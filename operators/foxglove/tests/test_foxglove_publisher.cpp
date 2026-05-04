/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include <fmt/format.h>
#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

#include "foxglove_publisher.hpp"

namespace holoscan::ops {
namespace {

constexpr uint64_t kSyntheticCaptureTimestampNs = 1'725'000'000'123'456'789ULL;

struct McapChannel {
  uint16_t id = 0;
  std::string topic;
  std::string encoding;
};

struct McapMessage {
  uint16_t channel_id = 0;
  uint64_t log_time = 0;
  std::vector<std::byte> data;
};

struct ParsedMcap {
  std::unordered_map<uint16_t, McapChannel> channels;
  std::vector<McapMessage> messages;
};

void require_bytes(size_t offset, size_t count, size_t size) {
  if (offset > size || count > size - offset) {
    throw std::runtime_error("Unexpected end of MCAP data");
  }
}

uint16_t read_u16(const uint8_t* data, size_t size, size_t& offset) {
  require_bytes(offset, sizeof(uint16_t), size);
  const uint16_t value = static_cast<uint16_t>(data[offset]) |
                         (static_cast<uint16_t>(data[offset + 1]) << 8);
  offset += sizeof(uint16_t);
  return value;
}

uint32_t read_u32(const uint8_t* data, size_t size, size_t& offset) {
  require_bytes(offset, sizeof(uint32_t), size);
  uint32_t value = 0;
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    value |= static_cast<uint32_t>(data[offset + i]) << (i * 8);
  }
  offset += sizeof(uint32_t);
  return value;
}

uint64_t read_u64(const uint8_t* data, size_t size, size_t& offset) {
  require_bytes(offset, sizeof(uint64_t), size);
  uint64_t value = 0;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    value |= static_cast<uint64_t>(data[offset + i]) << (i * 8);
  }
  offset += sizeof(uint64_t);
  return value;
}

std::string read_string(const uint8_t* data, size_t size, size_t& offset) {
  const auto length = read_u32(data, size, offset);
  require_bytes(offset, length, size);
  std::string value(reinterpret_cast<const char*>(data + offset), length);
  offset += length;
  return value;
}

void skip_string_map(const uint8_t* data, size_t size, size_t& offset) {
  const auto byte_length = read_u32(data, size, offset);
  require_bytes(offset, byte_length, size);
  offset += byte_length;
}

bool points_at_magic(const uint8_t* data, size_t size, size_t offset) {
  constexpr std::array<uint8_t, 8> kMagic{0x89, 'M', 'C', 'A', 'P', '0', '\r', '\n'};
  return offset + kMagic.size() <= size &&
         std::memcmp(data + offset, kMagic.data(), kMagic.size()) == 0;
}

void parse_mcap_records(const uint8_t* data, size_t size, ParsedMcap& parsed) {
  constexpr uint8_t kChannelRecord = 0x04;
  constexpr uint8_t kMessageRecord = 0x05;
  constexpr uint8_t kChunkRecord = 0x06;

  size_t offset = points_at_magic(data, size, 0) ? 8 : 0;
  while (offset + 9 <= size) {
    if (points_at_magic(data, size, offset)) {
      break;
    }

    const auto opcode = data[offset++];
    const auto length = read_u64(data, size, offset);
    require_bytes(offset, static_cast<size_t>(length), size);
    const auto record = data + offset;
    const auto record_size = static_cast<size_t>(length);

    size_t record_offset = 0;
    if (opcode == kChannelRecord) {
      McapChannel channel;
      channel.id = read_u16(record, record_size, record_offset);
      static_cast<void>(read_u16(record, record_size, record_offset));
      channel.topic = read_string(record, record_size, record_offset);
      channel.encoding = read_string(record, record_size, record_offset);
      skip_string_map(record, record_size, record_offset);
      parsed.channels[channel.id] = std::move(channel);
    } else if (opcode == kMessageRecord) {
      McapMessage message;
      message.channel_id = read_u16(record, record_size, record_offset);
      static_cast<void>(read_u32(record, record_size, record_offset));
      message.log_time = read_u64(record, record_size, record_offset);
      static_cast<void>(read_u64(record, record_size, record_offset));
      require_bytes(record_offset, record_size - record_offset, record_size);
      message.data.resize(record_size - record_offset);
      std::memcpy(message.data.data(), record + record_offset, message.data.size());
      parsed.messages.push_back(std::move(message));
    } else if (opcode == kChunkRecord) {
      static_cast<void>(read_u64(record, record_size, record_offset));
      static_cast<void>(read_u64(record, record_size, record_offset));
      static_cast<void>(read_u64(record, record_size, record_offset));
      static_cast<void>(read_u32(record, record_size, record_offset));
      const auto compression = read_string(record, record_size, record_offset);
      const auto records_length = read_u64(record, record_size, record_offset);
      require_bytes(record_offset, static_cast<size_t>(records_length), record_size);
      if (!compression.empty()) {
        throw std::runtime_error("Compressed MCAP chunks are not supported by this test reader");
      }
      parse_mcap_records(record + record_offset, static_cast<size_t>(records_length), parsed);
    }

    offset += record_size;
  }
}

ParsedMcap parse_mcap_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Could not open MCAP file");
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  ParsedMcap parsed;
  parse_mcap_records(bytes.data(), bytes.size(), parsed);
  return parsed;
}

std::filesystem::path temporary_mcap_path(const std::string& name) {
  return std::filesystem::temp_directory_path() /
         fmt::format("{}_{}.mcap", name, ::getpid());
}

class SyntheticImageSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SyntheticImageSourceOp)

  SyntheticImageSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_,
               "allocator",
               "Allocator",
               "Allocator for the synthetic image tensor",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("allocator")));
    spec.output<gxf::Entity>("output");
  }

  void initialize() override {
    add_arg(allocator_.default_value());
    Operator::initialize();
  }

  void compute([[maybe_unused]] InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto tensor =
        static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("image").value();
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());

    tensor->reshape<uint8_t>(nvidia::gxf::Shape({2, 2, 4}),
                             nvidia::gxf::MemoryStorageType::kHost,
                             gxf_allocator.value());

    const uint8_t pixels[] = {255, 0,   0,   255, 0,   255, 0,   255,
                              0,   0,   255, 255, 255, 255, 255, 255};
    std::memcpy(tensor->pointer(), pixels, sizeof(pixels));

    auto meta = metadata();
    meta->set("acquisition_timestamp_ns", kSyntheticCaptureTimestampNs);
    meta->set("frame_index", uint64_t{7});
    meta->set("sequence_id", uint64_t{11});

    op_output.emit(entity, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

class FoxglovePublisherMcapApp : public holoscan::Application {
 public:
  explicit FoxglovePublisherMcapApp(std::string mcap_path) : mcap_path_(std::move(mcap_path)) {
    enable_metadata(true);
  }

  void compose() override {
    auto source = make_operator<SyntheticImageSourceOp>(
        "source", make_condition<holoscan::CountCondition>(1));
    auto foxglove = make_operator<FoxglovePublisherOp>(
        "foxglove",
        Arg("port", uint16_t{0}),
        Arg("enable_mcap", true),
        Arg("mcap_path", mcap_path_),
        Arg("mcap_compression", std::string("none")),
        Arg("image_topic", std::string("/video")),
        Arg("image_frame_id", std::string("endoscope")),
        Arg("image_tensor_name", std::string("image")),
        Arg("image_encoding", std::string("rgba8")));

    add_flow(source, foxglove, {{"output", "image"}});
  }

 private:
  std::string mcap_path_;
};

ParsedMcap run_fragment_mcap_test(const std::string& name) {
  const auto path = temporary_mcap_path(name);
  std::filesystem::remove(path);

  auto application = holoscan::make_application<FoxglovePublisherMcapApp>(path.string());
  EXPECT_NO_THROW(application->run());
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_GT(std::filesystem::file_size(path), 0U);

  auto parsed = parse_mcap_file(path);
  std::filesystem::remove(path);
  return parsed;
}

}  // namespace

TEST(FoxglovePublisherHelpers, InferImageStep) {
  EXPECT_EQ(infer_image_step(640, "mono8"), 640u);
  EXPECT_EQ(infer_image_step(640, "rgba8"), 2560u);
  EXPECT_EQ(infer_image_step(640, "rgb8"), 1920u);
  EXPECT_EQ(infer_image_step(640, "mono16"), 1280u);
  EXPECT_EQ(infer_image_step(640, "32FC1"), 2560u);
}

TEST(FoxglovePublisherHelpers, ByteVectorCopy) {
  const uint8_t input[] = {1, 2, 3, 4};
  const auto bytes = to_byte_vector(input, sizeof(input));
  ASSERT_EQ(bytes.size(), sizeof(input));
  EXPECT_EQ(static_cast<uint8_t>(bytes[0]), 1);
  EXPECT_EQ(static_cast<uint8_t>(bytes[3]), 4);
}

TEST(FoxglovePublisherHelpers, ResolveTimestampUsesEpochNanosecondsForZero) {
  const auto before = now_epoch_ns();
  const auto resolved = resolve_timestamp_ns(0);
  const auto after = now_epoch_ns();

  EXPECT_GE(resolved, before);
  EXPECT_LE(resolved, after);
  EXPECT_EQ(resolve_timestamp_ns(1234), 1234ULL);
}

TEST(FoxglovePublisherHelpers, BatchCanCarryMultipleModalities) {
  FoxgloveBatch batch;
  FoxgloveImage color;
  color.topic = "/camera/image";
  color.width = 2;
  color.height = 1;
  color.encoding = "rgba8";
  color.step = infer_image_step(color.width, color.encoding);
  color.data.resize(color.step * color.height);

  FoxgloveImage depth;
  depth.topic = "/camera/depth";
  depth.width = 2;
  depth.height = 1;
  depth.encoding = "mono16";
  depth.step = infer_image_step(depth.width, depth.encoding);
  depth.data.resize(depth.step * depth.height);

  batch.images.push_back(color);
  batch.images.push_back(depth);

  ASSERT_EQ(batch.images.size(), 2U);
  EXPECT_EQ(batch.images[0].topic, "/camera/image");
  EXPECT_EQ(batch.images[1].encoding, "mono16");
}

TEST(FoxglovePublisherHelpers, BatchCanCarryInferenceArtifacts) {
  FoxgloveBatch batch;
  FoxgloveImage mask;
  mask.topic = "/segmentation";
  mask.width = 4;
  mask.height = 3;
  mask.encoding = "mono8";
  mask.step = infer_image_step(mask.width, mask.encoding);
  mask.data.resize(mask.step * mask.height);

  FoxgloveImageAnnotations detections;
  detections.topic = "/detections";
  FoxgloveBox2D box;
  box.x = 1.0;
  box.y = 2.0;
  box.width = 3.0;
  box.height = 4.0;
  box.label = "instrument";
  box.confidence = 0.875;
  detections.boxes.push_back(box);

  batch.images.push_back(mask);
  batch.annotations.push_back(detections);

  ASSERT_EQ(batch.images.size(), 1U);
  ASSERT_EQ(batch.annotations.size(), 1U);
  EXPECT_EQ(batch.images[0].topic, "/segmentation");
  EXPECT_EQ(batch.annotations[0].boxes[0].label, "instrument");

  FoxglovePointsAnnotation feature_points;
  feature_points.type = foxglove::messages::PointsAnnotation::PointsAnnotationType::POINTS;
  feature_points.label = "feature_points";
  feature_points.points.push_back({10.0, 12.0, 0.95, "feature_0"});
  batch.annotations[0].point_sets.push_back(feature_points);
  ASSERT_EQ(batch.annotations[0].point_sets.size(), 1U);
  EXPECT_EQ(batch.annotations[0].point_sets[0].points[0].label, "feature_0");
}

TEST(FoxglovePublisherHelpers, BatchCanCarryPointCloudAndState) {
  FoxgloveBatch batch;

  FoxglovePointCloud cloud;
  cloud.topic = "/depth/points";
  cloud.frame_id = "camera";
  cloud.point_stride = 12;
  cloud.fields.resize(3);
  cloud.fields[0].name = "x";
  cloud.fields[0].offset = 0;
  cloud.fields[0].type = foxglove::messages::PackedElementField::NumericType::FLOAT32;
  cloud.fields[1].name = "y";
  cloud.fields[1].offset = 4;
  cloud.fields[1].type = foxglove::messages::PackedElementField::NumericType::FLOAT32;
  cloud.fields[2].name = "z";
  cloud.fields[2].offset = 8;
  cloud.fields[2].type = foxglove::messages::PackedElementField::NumericType::FLOAT32;
  cloud.data.resize(24);

  FoxgloveKeyValue temperature;
  temperature.topic = "/system/tegrastats";
  temperature.key = "thermal.cpu-thermal_celsius";
  temperature.value = "48.125";

  batch.point_clouds.push_back(cloud);
  batch.key_values.push_back(temperature);

  ASSERT_EQ(batch.point_clouds.size(), 1U);
  ASSERT_EQ(batch.key_values.size(), 1U);
  EXPECT_EQ(batch.point_clouds[0].fields[2].name, "z");
  EXPECT_EQ(batch.key_values[0].key, "thermal.cpu-thermal_celsius");
}

TEST(FoxglovePublisherHelpers, BatchCanCarryCompressedVideoAndFrameTransforms) {
  FoxgloveBatch batch;

  FoxgloveCompressedVideo video;
  video.topic = "/video/compressed";
  video.frame_id = "endoscope";
  video.format = "h264";
  video.timestamp_ns = 42;
  video.data = to_byte_vector(reinterpret_cast<const uint8_t*>("nal"), 3);

  FoxgloveFrameTransform transform;
  transform.topic = "/tf";
  transform.parent_frame_id = "world";
  transform.child_frame_id = "endoscope";
  transform.timestamp_ns = 42;
  transform.translation = {1.0, 2.0, 3.0};
  transform.rotation = {0.0, 0.0, 0.0, 1.0};

  batch.compressed_videos.push_back(video);
  batch.frame_transforms.push_back(transform);

  ASSERT_EQ(batch.compressed_videos.size(), 1U);
  ASSERT_EQ(batch.frame_transforms.size(), 1U);
  EXPECT_EQ(batch.compressed_videos[0].format, "h264");
  EXPECT_EQ(batch.compressed_videos[0].data.size(), 3U);
  EXPECT_EQ(batch.frame_transforms[0].child_frame_id, "endoscope");
  EXPECT_EQ(batch.frame_transforms[0].translation[2], 3.0);
}

TEST(FoxglovePublisherIntegration, FragmentWritesImageMcapWithCaptureTimestamp) {
  const auto parsed = run_fragment_mcap_test("foxglove_timestamp_lineage");

  uint16_t video_channel_id = 0;
  for (const auto& [channel_id, channel] : parsed.channels) {
    if (channel.topic == "/video") {
      video_channel_id = channel_id;
      EXPECT_EQ(channel.encoding, "protobuf");
      break;
    }
  }
  ASSERT_NE(video_channel_id, 0U);

  size_t video_messages = 0;
  for (const auto& message : parsed.messages) {
    if (message.channel_id == video_channel_id) {
      ++video_messages;
      EXPECT_EQ(message.log_time, kSyntheticCaptureTimestampNs);
      EXPECT_GT(message.data.size(), 0U);
    }
  }
  EXPECT_EQ(video_messages, 1U);
}

TEST(FoxglovePublisherIntegration, McapRoundTripContainsMetadataSidecars) {
  const auto parsed = run_fragment_mcap_test("foxglove_mcap_roundtrip");

  std::unordered_map<std::string, size_t> message_counts;
  for (const auto& message : parsed.messages) {
    const auto channel = parsed.channels.find(message.channel_id);
    ASSERT_NE(channel, parsed.channels.end());
    ++message_counts[channel->second.topic];
  }

  EXPECT_EQ(message_counts["/video"], 1U);
  EXPECT_GE(message_counts["/metadata"], 2U);
}

}  // namespace holoscan::ops
