/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cuda_runtime_api.h>
#include <foxglove/mcap.hpp>
#include <foxglove/messages.hpp>
#include <foxglove/service.hpp>
#include <foxglove/websocket.hpp>
#include <gxf/multimedia/video.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/domain/tensor_map.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

struct FoxgloveImage {
  std::string topic;
  std::string frame_id = "camera";
  std::string encoding = "rgba8";
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t step = 0;
  uint64_t timestamp_ns = 0;
  std::vector<std::byte> data;
};

struct FoxgloveCompressedVideo {
  std::string topic;
  std::string frame_id = "camera";
  std::string format = "h264";
  uint64_t timestamp_ns = 0;
  std::vector<std::byte> data;
};

struct FoxgloveCameraCalibration {
  std::string topic;
  std::string frame_id = "camera";
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t timestamp_ns = 0;
  std::string distortion_model;
  std::vector<double> distortion;
  std::array<double, 9> k{};
  std::array<double, 9> r{};
  std::array<double, 12> p{};
};

struct FoxgloveBox2D {
  double x = 0.0;
  double y = 0.0;
  double width = 0.0;
  double height = 0.0;
  std::string label;
  double confidence = -1.0;
};

struct FoxgloveText {
  double x = 0.0;
  double y = 0.0;
  std::string text;
  double font_size = 16.0;
};

struct FoxglovePoint2D {
  double x = 0.0;
  double y = 0.0;
  double confidence = -1.0;
  std::string label;
};

struct FoxglovePointsAnnotation {
  foxglove::messages::PointsAnnotation::PointsAnnotationType type =
      foxglove::messages::PointsAnnotation::PointsAnnotationType::POINTS;
  std::vector<FoxglovePoint2D> points;
  std::string label;
  double thickness = 4.0;
  std::array<double, 4> color{0.1, 0.8, 1.0, 1.0};
};

struct FoxgloveImageAnnotations {
  std::string topic;
  uint64_t timestamp_ns = 0;
  std::vector<FoxgloveBox2D> boxes;
  std::vector<FoxglovePointsAnnotation> point_sets;
  std::vector<FoxgloveText> texts;
};

struct FoxglovePointCloud {
  std::string topic;
  std::string frame_id = "sensor";
  uint64_t timestamp_ns = 0;
  uint32_t point_stride = 0;
  std::vector<foxglove::messages::PackedElementField> fields;
  std::vector<std::byte> data;
};

struct FoxgloveKeyValue {
  std::string topic = "/state";
  std::string key;
  std::string value;
  uint64_t timestamp_ns = 0;
};

struct FoxgloveFrameTransform {
  std::string topic = "/tf";
  std::string parent_frame_id = "world";
  std::string child_frame_id = "sensor";
  uint64_t timestamp_ns = 0;
  std::array<double, 3> translation{0.0, 0.0, 0.0};
  std::array<double, 4> rotation{0.0, 0.0, 0.0, 1.0};
};

struct FoxgloveBatch {
  std::vector<FoxgloveImage> images;
  std::vector<FoxgloveCompressedVideo> compressed_videos;
  std::vector<FoxgloveCameraCalibration> calibrations;
  std::vector<FoxgloveImageAnnotations> annotations;
  std::vector<FoxglovePointCloud> point_clouds;
  std::vector<FoxgloveFrameTransform> frame_transforms;
  std::vector<FoxgloveKeyValue> key_values;
};

class PinnedHostBufferPool {
 public:
  PinnedHostBufferPool() = default;
  ~PinnedHostBufferPool();

  PinnedHostBufferPool(const PinnedHostBufferPool&) = delete;
  PinnedHostBufferPool& operator=(const PinnedHostBufferPool&) = delete;
  PinnedHostBufferPool(PinnedHostBufferPool&&) = delete;
  PinnedHostBufferPool& operator=(PinnedHostBufferPool&&) = delete;

  std::byte* acquire(size_t size);
  void clear();

 private:
  struct Buffer {
    std::byte* data = nullptr;
    size_t capacity = 0;
  };

  static size_t size_class(size_t size);

  std::vector<Buffer> buffers_;
};

std::vector<std::byte> to_byte_vector(const uint8_t* data, size_t size);
uint64_t now_epoch_ns();
uint64_t resolve_timestamp_ns(uint64_t timestamp_ns);
uint32_t infer_image_step(uint32_t width, const std::string& encoding);

class FoxglovePublisherOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FoxglovePublisherOp)

  FoxglovePublisherOp() = default;
  ~FoxglovePublisherOp() override = default;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  using RawImageChannelMap = std::unordered_map<std::string, foxglove::messages::RawImageChannel>;
  using CompressedVideoChannelMap =
      std::unordered_map<std::string, foxglove::messages::CompressedVideoChannel>;
  using CalibrationChannelMap =
      std::unordered_map<std::string, foxglove::messages::CameraCalibrationChannel>;
  using AnnotationChannelMap =
      std::unordered_map<std::string, foxglove::messages::ImageAnnotationsChannel>;
  using PointCloudChannelMap =
      std::unordered_map<std::string, foxglove::messages::PointCloudChannel>;
  using FrameTransformChannelMap =
      std::unordered_map<std::string, foxglove::messages::FrameTransformChannel>;
  using KeyValueChannelMap =
      std::unordered_map<std::string, foxglove::messages::KeyValuePairChannel>;
  using ParameterValue = std::variant<std::monostate, bool, std::string, double, int64_t>;

  struct PendingParameterUpdate {
    std::string name;
    ParameterValue value;
  };

  uint64_t publish_image(const FoxgloveImage& image);
  uint64_t publish_compressed_video(const FoxgloveCompressedVideo& video);
  uint64_t publish_calibration(const FoxgloveCameraCalibration& calibration);
  uint64_t publish_annotations(const FoxgloveImageAnnotations& annotations);
  uint64_t publish_point_cloud(const FoxglovePointCloud& point_cloud);
  uint64_t publish_frame_transform(const FoxgloveFrameTransform& transform);
  uint64_t publish_key_value(const FoxgloveKeyValue& key_value);
  void open_mcap_writer(const std::string& path);
  void close_mcap_writer();
  std::string snapshot_mcap_path() const;
  void register_services();
  void precreate_channels();
  FoxgloveImage image_from_entity(gxf::Entity entity, cudaStream_t stream);
  FoxgloveImage image_from_tensor_map(const TensorMap& tensors, cudaStream_t stream);
  std::vector<foxglove::Parameter> foxglove_parameters(
      const std::vector<std::string_view>& names = {});
  std::vector<foxglove::Parameter> enqueue_foxglove_parameter_updates(
      const std::vector<foxglove::ParameterView>& params);
  void apply_pending_parameter_updates();
  bool drop_when_unsubscribed() const;
  bool should_publish_raw_image(const std::string& topic);

  foxglove::messages::RawImageChannel& raw_image_channel(const std::string& topic);
  foxglove::messages::CompressedVideoChannel& compressed_video_channel(const std::string& topic);
  foxglove::messages::CameraCalibrationChannel& calibration_channel(const std::string& topic);
  foxglove::messages::ImageAnnotationsChannel& annotation_channel(const std::string& topic);
  foxglove::messages::PointCloudChannel& point_cloud_channel(const std::string& topic);
  foxglove::messages::FrameTransformChannel& frame_transform_channel(const std::string& topic);
  foxglove::messages::KeyValuePairChannel& key_value_channel(const std::string& topic);

  Parameter<std::string> bind_address_;
  Parameter<uint16_t> port_;
  Parameter<std::string> server_name_;
  Parameter<bool> publish_server_time_;
  Parameter<bool> drop_when_unsubscribed_;
  Parameter<bool> enable_mcap_;
  Parameter<std::string> mcap_path_;
  Parameter<std::string> mcap_compression_;
  Parameter<std::string> timestamp_metadata_keys_;
  Parameter<std::string> mutable_parameters_;
  Parameter<std::string> image_topic_;
  Parameter<std::string> image_frame_id_;
  Parameter<std::string> image_tensor_name_;
  Parameter<std::string> image_encoding_;
  Parameter<uint32_t> image_width_;
  Parameter<uint32_t> image_height_;
  Parameter<uint32_t> image_step_;
  Parameter<bool> image_prefer_video_buffer_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  mutable std::mutex mcap_mutex_;
  mutable std::mutex parameter_mutex_;
  foxglove::Context context_;
  std::optional<foxglove::WebSocketServer> server_;
  std::optional<foxglove::McapWriter> mcap_writer_;
  std::vector<foxglove::ServiceHandler> service_handlers_;
  RawImageChannelMap raw_image_channels_;
  CompressedVideoChannelMap compressed_video_channels_;
  CalibrationChannelMap calibration_channels_;
  AnnotationChannelMap annotation_channels_;
  PointCloudChannelMap point_cloud_channels_;
  FrameTransformChannelMap frame_transform_channels_;
  KeyValueChannelMap key_value_channels_;
  std::vector<PendingParameterUpdate> pending_parameter_updates_;
  PinnedHostBufferPool pinned_host_pool_;
};

class FoxgloveTensorAdapterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FoxgloveTensorAdapterOp)

  FoxgloveTensorAdapterOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  FoxgloveImage image_from_video_buffer(const nvidia::gxf::VideoBuffer& buffer,
                                        cudaStream_t stream) const;
  FoxgloveImage image_from_tensor(const Tensor& tensor, cudaStream_t stream) const;

  Parameter<std::string> topic_;
  Parameter<std::string> frame_id_;
  Parameter<std::string> tensor_name_;
  Parameter<std::string> encoding_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> step_;
  Parameter<bool> prefer_video_buffer_;
  Parameter<std::string> timestamp_metadata_keys_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  mutable PinnedHostBufferPool pinned_host_pool_;
  bool topic_logged_ = false;
};

class FoxgloveDetectionAdapterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FoxgloveDetectionAdapterOp)

  FoxgloveDetectionAdapterOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  std::vector<std::string> labels() const;

  Parameter<std::string> annotation_topic_;
  Parameter<std::string> boxes_tensor_;
  Parameter<std::string> scores_tensor_;
  Parameter<std::string> labels_tensor_;
  Parameter<std::string> combined_tensor_;
  Parameter<std::string> combined_format_;
  Parameter<std::string> box_format_;
  Parameter<std::string> label_map_;
  Parameter<uint32_t> image_width_;
  Parameter<uint32_t> image_height_;
  Parameter<double> score_threshold_;
  Parameter<bool> normalized_coordinates_;
  Parameter<bool> clamp_to_image_;
  Parameter<std::string> timestamp_metadata_keys_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  mutable PinnedHostBufferPool pinned_host_pool_;
  bool topic_logged_ = false;
};

class FoxgloveSegmentationMaskAdapterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FoxgloveSegmentationMaskAdapterOp)

  FoxgloveSegmentationMaskAdapterOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::string> topic_;
  Parameter<std::string> frame_id_;
  Parameter<std::string> tensor_name_;
  Parameter<std::string> encoding_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<uint32_t> step_;
  Parameter<std::string> timestamp_metadata_keys_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  mutable PinnedHostBufferPool pinned_host_pool_;
  bool topic_logged_ = false;
};

class FoxgloveCompressedVideoAdapterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FoxgloveCompressedVideoAdapterOp)

  FoxgloveCompressedVideoAdapterOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::string> topic_;
  Parameter<std::string> frame_id_;
  Parameter<std::string> tensor_name_;
  Parameter<std::string> format_;
  Parameter<std::string> timestamp_metadata_keys_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  mutable PinnedHostBufferPool pinned_host_pool_;
  bool topic_logged_ = false;
};

class FoxglovePoseAdapterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FoxglovePoseAdapterOp)

  FoxglovePoseAdapterOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::string> topic_;
  Parameter<std::string> tensor_name_;
  Parameter<std::string> parent_frame_id_;
  Parameter<std::string> child_frame_id_;
  Parameter<std::string> format_;
  Parameter<std::string> timestamp_metadata_keys_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  mutable PinnedHostBufferPool pinned_host_pool_;
  bool topic_logged_ = false;
};

}  // namespace holoscan::ops
