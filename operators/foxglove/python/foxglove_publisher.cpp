/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../foxglove_publisher.hpp"
#include "../../operator_util.hpp"
#include "foxglove_publisher_pydoc.hpp"

#include <cstring>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#if __has_include(<holoscan/python/core/emitter_receiver_registry.hpp>)
#include <holoscan/python/core/emitter_receiver_registry.hpp>
#define HOLOHUB_FOXGLOVE_HAS_EMITTER_RECEIVER_REGISTRY 1
#else
#define HOLOHUB_FOXGLOVE_HAS_EMITTER_RECEIVER_REGISTRY 0
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using pybind11::literals::operator""_a;
using std::string_literals::operator""s;

namespace holoscan::ops {

class PyFoxglovePublisherOp : public FoxglovePublisherOp {
 public:
  using FoxglovePublisherOp::FoxglovePublisherOp;

  PyFoxglovePublisherOp(Fragment* fragment,
                        const py::args& args,
                        const std::string& bind_address = "0.0.0.0",
                        uint16_t port = 8765,
                        const std::string& server_name = "Holoscan Foxglove",
                        bool publish_server_time = true,
                        bool drop_when_unsubscribed = true,
                        bool enable_mcap = false,
                        const std::string& mcap_path = "holoscan_foxglove.mcap",
                        const std::string& mcap_compression = "zstd",
                        const std::string& timestamp_metadata_keys =
                            "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns",
                        const std::string& mutable_parameters = "",
                        const std::string& image_topic = "/image",
                        const std::string& image_frame_id = "camera",
                        const std::string& image_tensor_name = "",
                        const std::string& image_encoding = "",
                        uint32_t image_width = 0,
                        uint32_t image_height = 0,
                        uint32_t image_step = 0,
                        bool image_prefer_video_buffer = true,
                        std::shared_ptr<Allocator> allocator = nullptr,
                        const std::string& name = "foxglove_publisher")
      : FoxglovePublisherOp(ArgList{Arg{"bind_address", bind_address},
                                    Arg{"port", port},
                                    Arg{"server_name", server_name},
                                    Arg{"publish_server_time", publish_server_time},
                                    Arg{"drop_when_unsubscribed", drop_when_unsubscribed},
                                    Arg{"enable_mcap", enable_mcap},
                                    Arg{"mcap_path", mcap_path},
                                    Arg{"mcap_compression", mcap_compression},
                                    Arg{"timestamp_metadata_keys", timestamp_metadata_keys},
                                    Arg{"mutable_parameters", mutable_parameters},
                                    Arg{"image_topic", image_topic},
                                    Arg{"image_frame_id", image_frame_id},
                                    Arg{"image_tensor_name", image_tensor_name},
                                    Arg{"image_encoding", image_encoding},
                                    Arg{"image_width", image_width},
                                    Arg{"image_height", image_height},
                                    Arg{"image_step", image_step},
                                    Arg{"image_prefer_video_buffer",
                                        image_prefer_video_buffer},
                                    Arg{"allocator", allocator}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

class PyFoxgloveTensorAdapterOp : public FoxgloveTensorAdapterOp {
 public:
  using FoxgloveTensorAdapterOp::FoxgloveTensorAdapterOp;

  PyFoxgloveTensorAdapterOp(Fragment* fragment,
                            const py::args& args,
                            const std::string& topic = "/image",
                            const std::string& frame_id = "camera",
                            const std::string& tensor_name = "",
                            const std::string& encoding = "",
                            uint32_t width = 0,
                            uint32_t height = 0,
                            uint32_t step = 0,
                            bool prefer_video_buffer = true,
                            const std::string& timestamp_metadata_keys =
                                "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns",
                            std::shared_ptr<Allocator> allocator = nullptr,
                            const std::string& name = "foxglove_tensor_adapter")
      : FoxgloveTensorAdapterOp(ArgList{Arg{"topic", topic},
                                        Arg{"frame_id", frame_id},
                                        Arg{"tensor_name", tensor_name},
                                        Arg{"encoding", encoding},
                                        Arg{"width", width},
                                        Arg{"height", height},
                                        Arg{"step", step},
                                        Arg{"prefer_video_buffer", prefer_video_buffer},
                                        Arg{"timestamp_metadata_keys", timestamp_metadata_keys},
                                        Arg{"allocator", allocator}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

class PyFoxgloveDetectionAdapterOp : public FoxgloveDetectionAdapterOp {
 public:
  using FoxgloveDetectionAdapterOp::FoxgloveDetectionAdapterOp;

  PyFoxgloveDetectionAdapterOp(Fragment* fragment,
                               const py::args& args,
                               const std::string& annotation_topic = "/detections",
                               const std::string& boxes_tensor = "boxes",
                               const std::string& scores_tensor = "scores",
                               const std::string& labels_tensor = "labels",
                               const std::string& combined_tensor = "",
                               const std::string& combined_format = "xyxy_score_label",
                               const std::string& box_format = "xyxy",
                               const std::string& label_map = "",
                               uint32_t image_width = 0,
                               uint32_t image_height = 0,
                               double score_threshold = 0.25,
                               bool normalized_coordinates = false,
                               bool clamp_to_image = true,
                               const std::string& timestamp_metadata_keys =
                                   "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns",
                               std::shared_ptr<Allocator> allocator = nullptr,
                               const std::string& name = "foxglove_detection_adapter")
      : FoxgloveDetectionAdapterOp(ArgList{Arg{"annotation_topic", annotation_topic},
                                           Arg{"boxes_tensor", boxes_tensor},
                                           Arg{"scores_tensor", scores_tensor},
                                           Arg{"labels_tensor", labels_tensor},
                                           Arg{"combined_tensor", combined_tensor},
                                           Arg{"combined_format", combined_format},
                                           Arg{"box_format", box_format},
                                           Arg{"label_map", label_map},
                                           Arg{"image_width", image_width},
                                           Arg{"image_height", image_height},
                                           Arg{"score_threshold", score_threshold},
                                           Arg{"normalized_coordinates", normalized_coordinates},
                                           Arg{"clamp_to_image", clamp_to_image},
                                           Arg{"timestamp_metadata_keys",
                                               timestamp_metadata_keys},
                                           Arg{"allocator", allocator}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

class PyFoxgloveSegmentationMaskAdapterOp : public FoxgloveSegmentationMaskAdapterOp {
 public:
  using FoxgloveSegmentationMaskAdapterOp::FoxgloveSegmentationMaskAdapterOp;

  PyFoxgloveSegmentationMaskAdapterOp(Fragment* fragment,
                                      const py::args& args,
                                      const std::string& topic = "/segmentation",
                                      const std::string& frame_id = "camera",
                                      const std::string& tensor_name = "out_tensor",
                                      const std::string& encoding = "mono8",
                                      uint32_t width = 0,
                                      uint32_t height = 0,
                                      uint32_t step = 0,
                                      const std::string& timestamp_metadata_keys =
                                          "acquisition_timestamp_ns,timestamp_ns,"
                                          "sensor_timestamp_ns",
                                      std::shared_ptr<Allocator> allocator = nullptr,
                                      const std::string& name =
                                          "foxglove_segmentation_mask_adapter")
      : FoxgloveSegmentationMaskAdapterOp(ArgList{Arg{"topic", topic},
                                                  Arg{"frame_id", frame_id},
                                                  Arg{"tensor_name", tensor_name},
                                                  Arg{"encoding", encoding},
                                                  Arg{"width", width},
                                                  Arg{"height", height},
                                                  Arg{"step", step},
                                                  Arg{"timestamp_metadata_keys",
                                                      timestamp_metadata_keys},
                                                  Arg{"allocator", allocator}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

class PyFoxgloveCompressedVideoAdapterOp : public FoxgloveCompressedVideoAdapterOp {
 public:
  using FoxgloveCompressedVideoAdapterOp::FoxgloveCompressedVideoAdapterOp;

  PyFoxgloveCompressedVideoAdapterOp(
      Fragment* fragment,
      const py::args& args,
      const std::string& topic = "/video/compressed",
      const std::string& frame_id = "camera",
      const std::string& tensor_name = "",
      const std::string& format = "h264",
      const std::string& timestamp_metadata_keys =
          "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns",
      std::shared_ptr<Allocator> allocator = nullptr,
      const std::string& name = "foxglove_compressed_video_adapter")
      : FoxgloveCompressedVideoAdapterOp(ArgList{
            Arg{"topic", topic},
            Arg{"frame_id", frame_id},
            Arg{"tensor_name", tensor_name},
            Arg{"format", format},
            Arg{"timestamp_metadata_keys", timestamp_metadata_keys},
            Arg{"allocator", allocator}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

class PyFoxglovePoseAdapterOp : public FoxglovePoseAdapterOp {
 public:
  using FoxglovePoseAdapterOp::FoxglovePoseAdapterOp;

  PyFoxglovePoseAdapterOp(Fragment* fragment,
                          const py::args& args,
                          const std::string& topic = "/tf",
                          const std::string& tensor_name = "",
                          const std::string& parent_frame_id = "world",
                          const std::string& child_frame_id = "sensor",
                          const std::string& format = "matrix4x4",
                          const std::string& timestamp_metadata_keys =
                              "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns",
                          std::shared_ptr<Allocator> allocator = nullptr,
                          const std::string& name = "foxglove_pose_adapter")
      : FoxglovePoseAdapterOp(ArgList{Arg{"topic", topic},
                                      Arg{"tensor_name", tensor_name},
                                      Arg{"parent_frame_id", parent_frame_id},
                                      Arg{"child_frame_id", child_frame_id},
                                      Arg{"format", format},
                                      Arg{"timestamp_metadata_keys", timestamp_metadata_keys},
                                      Arg{"allocator", allocator}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

PYBIND11_MODULE(_foxglove, m) {
  m.doc() = "Holoscan Foxglove operator bindings";

  py::class_<FoxgloveImage, std::shared_ptr<FoxgloveImage>>(m, "FoxgloveImage")
      .def(py::init<>())
      .def_readwrite("topic", &FoxgloveImage::topic)
      .def_readwrite("frame_id", &FoxgloveImage::frame_id)
      .def_readwrite("encoding", &FoxgloveImage::encoding)
      .def_readwrite("width", &FoxgloveImage::width)
      .def_readwrite("height", &FoxgloveImage::height)
      .def_readwrite("step", &FoxgloveImage::step)
      .def_readwrite("timestamp_ns", &FoxgloveImage::timestamp_ns)
      .def_property(
          "data",
          [](const FoxgloveImage& image) {
            return py::bytes(reinterpret_cast<const char*>(image.data.data()), image.data.size());
          },
          [](FoxgloveImage& image, py::bytes bytes) {
            const auto text = static_cast<std::string>(bytes);
            image.data.resize(text.size());
            std::memcpy(image.data.data(), text.data(), text.size());
          });

  py::class_<FoxgloveCompressedVideo, std::shared_ptr<FoxgloveCompressedVideo>>(
      m, "FoxgloveCompressedVideo")
      .def(py::init<>())
      .def_readwrite("topic", &FoxgloveCompressedVideo::topic)
      .def_readwrite("frame_id", &FoxgloveCompressedVideo::frame_id)
      .def_readwrite("format", &FoxgloveCompressedVideo::format)
      .def_readwrite("timestamp_ns", &FoxgloveCompressedVideo::timestamp_ns)
      .def_property(
          "data",
          [](const FoxgloveCompressedVideo& video) {
            return py::bytes(reinterpret_cast<const char*>(video.data.data()), video.data.size());
          },
          [](FoxgloveCompressedVideo& video, py::bytes bytes) {
            const auto text = static_cast<std::string>(bytes);
            video.data.resize(text.size());
            std::memcpy(video.data.data(), text.data(), text.size());
          });

  py::class_<FoxgloveCameraCalibration, std::shared_ptr<FoxgloveCameraCalibration>>(
      m, "FoxgloveCameraCalibration")
      .def(py::init<>())
      .def_readwrite("topic", &FoxgloveCameraCalibration::topic)
      .def_readwrite("frame_id", &FoxgloveCameraCalibration::frame_id)
      .def_readwrite("width", &FoxgloveCameraCalibration::width)
      .def_readwrite("height", &FoxgloveCameraCalibration::height)
      .def_readwrite("timestamp_ns", &FoxgloveCameraCalibration::timestamp_ns)
      .def_readwrite("distortion_model", &FoxgloveCameraCalibration::distortion_model)
      .def_readwrite("distortion", &FoxgloveCameraCalibration::distortion)
      .def_readwrite("k", &FoxgloveCameraCalibration::k)
      .def_readwrite("r", &FoxgloveCameraCalibration::r)
      .def_readwrite("p", &FoxgloveCameraCalibration::p);

  py::class_<FoxgloveBox2D, std::shared_ptr<FoxgloveBox2D>>(m, "FoxgloveBox2D")
      .def(py::init<>())
      .def_readwrite("x", &FoxgloveBox2D::x)
      .def_readwrite("y", &FoxgloveBox2D::y)
      .def_readwrite("width", &FoxgloveBox2D::width)
      .def_readwrite("height", &FoxgloveBox2D::height)
      .def_readwrite("label", &FoxgloveBox2D::label)
      .def_readwrite("confidence", &FoxgloveBox2D::confidence);

  py::class_<FoxgloveText, std::shared_ptr<FoxgloveText>>(m, "FoxgloveText")
      .def(py::init<>())
      .def_readwrite("x", &FoxgloveText::x)
      .def_readwrite("y", &FoxgloveText::y)
      .def_readwrite("text", &FoxgloveText::text)
      .def_readwrite("font_size", &FoxgloveText::font_size);

  py::enum_<foxglove::messages::PointsAnnotation::PointsAnnotationType>(
      m, "PointsAnnotationType")
      .value("UNKNOWN", foxglove::messages::PointsAnnotation::PointsAnnotationType::UNKNOWN)
      .value("POINTS", foxglove::messages::PointsAnnotation::PointsAnnotationType::POINTS)
      .value("LINE_LOOP", foxglove::messages::PointsAnnotation::PointsAnnotationType::LINE_LOOP)
      .value("LINE_STRIP", foxglove::messages::PointsAnnotation::PointsAnnotationType::LINE_STRIP)
      .value("LINE_LIST", foxglove::messages::PointsAnnotation::PointsAnnotationType::LINE_LIST);

  py::class_<FoxglovePoint2D, std::shared_ptr<FoxglovePoint2D>>(m, "FoxglovePoint2D")
      .def(py::init<>())
      .def_readwrite("x", &FoxglovePoint2D::x)
      .def_readwrite("y", &FoxglovePoint2D::y)
      .def_readwrite("confidence", &FoxglovePoint2D::confidence)
      .def_readwrite("label", &FoxglovePoint2D::label);

  py::class_<FoxglovePointsAnnotation, std::shared_ptr<FoxglovePointsAnnotation>>(
      m, "FoxglovePointsAnnotation")
      .def(py::init<>())
      .def_readwrite("type", &FoxglovePointsAnnotation::type)
      .def_readwrite("points", &FoxglovePointsAnnotation::points)
      .def_readwrite("label", &FoxglovePointsAnnotation::label)
      .def_readwrite("thickness", &FoxglovePointsAnnotation::thickness)
      .def_readwrite("color", &FoxglovePointsAnnotation::color);

  py::class_<FoxgloveImageAnnotations, std::shared_ptr<FoxgloveImageAnnotations>>(
      m, "FoxgloveImageAnnotations")
      .def(py::init<>())
      .def_readwrite("topic", &FoxgloveImageAnnotations::topic)
      .def_readwrite("timestamp_ns", &FoxgloveImageAnnotations::timestamp_ns)
      .def_readwrite("boxes", &FoxgloveImageAnnotations::boxes)
      .def_readwrite("point_sets", &FoxgloveImageAnnotations::point_sets)
      .def_readwrite("texts", &FoxgloveImageAnnotations::texts);

  py::class_<foxglove::messages::PackedElementField> packed_field(m, "PackedElementField");
  py::enum_<foxglove::messages::PackedElementField::NumericType>(packed_field, "NumericType")
      .value("UNKNOWN", foxglove::messages::PackedElementField::NumericType::UNKNOWN)
      .value("UINT8", foxglove::messages::PackedElementField::NumericType::UINT8)
      .value("INT8", foxglove::messages::PackedElementField::NumericType::INT8)
      .value("UINT16", foxglove::messages::PackedElementField::NumericType::UINT16)
      .value("INT16", foxglove::messages::PackedElementField::NumericType::INT16)
      .value("UINT32", foxglove::messages::PackedElementField::NumericType::UINT32)
      .value("INT32", foxglove::messages::PackedElementField::NumericType::INT32)
      .value("FLOAT32", foxglove::messages::PackedElementField::NumericType::FLOAT32)
      .value("FLOAT64", foxglove::messages::PackedElementField::NumericType::FLOAT64);
  packed_field.def(py::init<>())
      .def_readwrite("name", &foxglove::messages::PackedElementField::name)
      .def_readwrite("offset", &foxglove::messages::PackedElementField::offset)
      .def_readwrite("type", &foxglove::messages::PackedElementField::type);

  py::class_<FoxglovePointCloud, std::shared_ptr<FoxglovePointCloud>>(m, "FoxglovePointCloud")
      .def(py::init<>())
      .def_readwrite("topic", &FoxglovePointCloud::topic)
      .def_readwrite("frame_id", &FoxglovePointCloud::frame_id)
      .def_readwrite("timestamp_ns", &FoxglovePointCloud::timestamp_ns)
      .def_readwrite("point_stride", &FoxglovePointCloud::point_stride)
      .def_readwrite("fields", &FoxglovePointCloud::fields)
      .def_property(
          "data",
          [](const FoxglovePointCloud& cloud) {
            return py::bytes(reinterpret_cast<const char*>(cloud.data.data()), cloud.data.size());
          },
          [](FoxglovePointCloud& cloud, py::bytes bytes) {
            const auto text = static_cast<std::string>(bytes);
            cloud.data.resize(text.size());
            std::memcpy(cloud.data.data(), text.data(), text.size());
          });

  py::class_<FoxgloveKeyValue, std::shared_ptr<FoxgloveKeyValue>>(m, "FoxgloveKeyValue")
      .def(py::init<>())
      .def_readwrite("topic", &FoxgloveKeyValue::topic)
      .def_readwrite("key", &FoxgloveKeyValue::key)
      .def_readwrite("value", &FoxgloveKeyValue::value)
      .def_readwrite("timestamp_ns", &FoxgloveKeyValue::timestamp_ns);

  py::class_<FoxgloveFrameTransform, std::shared_ptr<FoxgloveFrameTransform>>(
      m, "FoxgloveFrameTransform")
      .def(py::init<>())
      .def_readwrite("topic", &FoxgloveFrameTransform::topic)
      .def_readwrite("parent_frame_id", &FoxgloveFrameTransform::parent_frame_id)
      .def_readwrite("child_frame_id", &FoxgloveFrameTransform::child_frame_id)
      .def_readwrite("timestamp_ns", &FoxgloveFrameTransform::timestamp_ns)
      .def_readwrite("translation", &FoxgloveFrameTransform::translation)
      .def_readwrite("rotation", &FoxgloveFrameTransform::rotation);

  py::class_<FoxgloveBatch, std::shared_ptr<FoxgloveBatch>>(m, "FoxgloveBatch")
      .def(py::init<>())
      .def_readwrite("images", &FoxgloveBatch::images)
      .def_readwrite("compressed_videos", &FoxgloveBatch::compressed_videos)
      .def_readwrite("calibrations", &FoxgloveBatch::calibrations)
      .def_readwrite("annotations", &FoxgloveBatch::annotations)
      .def_readwrite("point_clouds", &FoxgloveBatch::point_clouds)
      .def_readwrite("frame_transforms", &FoxgloveBatch::frame_transforms)
      .def_readwrite("key_values", &FoxgloveBatch::key_values);

  py::class_<FoxglovePublisherOp,
             PyFoxglovePublisherOp,
             Operator,
             std::shared_ptr<FoxglovePublisherOp>>(m, "FoxglovePublisherOp",
                                                   holoscan::doc::FoxglovePublisherOp::doc)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    uint16_t,
                    const std::string&,
                    bool,
                    bool,
                    bool,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "bind_address"_a = "0.0.0.0"s,
           "port"_a = 8765,
           "server_name"_a = "Holoscan Foxglove"s,
           "publish_server_time"_a = true,
           "drop_when_unsubscribed"_a = true,
           "enable_mcap"_a = false,
           "mcap_path"_a = "holoscan_foxglove.mcap"s,
           "mcap_compression"_a = "zstd"s,
           "timestamp_metadata_keys"_a =
               "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns"s,
           "mutable_parameters"_a = ""s,
           "image_topic"_a = "/image"s,
           "image_frame_id"_a = "camera"s,
           "image_tensor_name"_a = ""s,
           "image_encoding"_a = ""s,
           "image_width"_a = 0,
           "image_height"_a = 0,
           "image_step"_a = 0,
           "image_prefer_video_buffer"_a = true,
           "allocator"_a = std::shared_ptr<Allocator>(),
           "name"_a = "foxglove_publisher"s);

  py::class_<FoxgloveTensorAdapterOp,
             PyFoxgloveTensorAdapterOp,
             Operator,
             std::shared_ptr<FoxgloveTensorAdapterOp>>(m, "FoxgloveTensorAdapterOp",
                                                       holoscan::doc::FoxgloveTensorAdapterOp::doc)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    const std::string&,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "topic"_a = "/image"s,
           "frame_id"_a = "camera"s,
           "tensor_name"_a = ""s,
           "encoding"_a = ""s,
           "width"_a = 0,
           "height"_a = 0,
           "step"_a = 0,
           "prefer_video_buffer"_a = true,
           "timestamp_metadata_keys"_a =
               "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns"s,
           "allocator"_a = std::shared_ptr<Allocator>(),
           "name"_a = "foxglove_tensor_adapter"s);

  py::class_<FoxgloveDetectionAdapterOp,
             PyFoxgloveDetectionAdapterOp,
             Operator,
             std::shared_ptr<FoxgloveDetectionAdapterOp>>(
      m, "FoxgloveDetectionAdapterOp", holoscan::doc::FoxgloveDetectionAdapterOp::doc)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    double,
                    bool,
                    bool,
                    const std::string&,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "annotation_topic"_a = "/detections"s,
           "boxes_tensor"_a = "boxes"s,
           "scores_tensor"_a = "scores"s,
           "labels_tensor"_a = "labels"s,
           "combined_tensor"_a = ""s,
           "combined_format"_a = "xyxy_score_label"s,
           "box_format"_a = "xyxy"s,
           "label_map"_a = ""s,
           "image_width"_a = 0,
           "image_height"_a = 0,
           "score_threshold"_a = 0.25,
           "normalized_coordinates"_a = false,
           "clamp_to_image"_a = true,
           "timestamp_metadata_keys"_a =
               "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns"s,
           "allocator"_a = std::shared_ptr<Allocator>(),
           "name"_a = "foxglove_detection_adapter"s);

  py::class_<FoxgloveSegmentationMaskAdapterOp,
             PyFoxgloveSegmentationMaskAdapterOp,
             Operator,
             std::shared_ptr<FoxgloveSegmentationMaskAdapterOp>>(
      m,
      "FoxgloveSegmentationMaskAdapterOp",
      holoscan::doc::FoxgloveSegmentationMaskAdapterOp::doc)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "topic"_a = "/segmentation"s,
           "frame_id"_a = "camera"s,
           "tensor_name"_a = "out_tensor"s,
           "encoding"_a = "mono8"s,
           "width"_a = 0,
           "height"_a = 0,
           "step"_a = 0,
           "timestamp_metadata_keys"_a =
               "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns"s,
           "allocator"_a = std::shared_ptr<Allocator>(),
           "name"_a = "foxglove_segmentation_mask_adapter"s);

  py::class_<FoxgloveCompressedVideoAdapterOp,
             PyFoxgloveCompressedVideoAdapterOp,
             Operator,
             std::shared_ptr<FoxgloveCompressedVideoAdapterOp>>(
      m,
      "FoxgloveCompressedVideoAdapterOp",
      holoscan::doc::FoxgloveCompressedVideoAdapterOp::doc)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "topic"_a = "/video/compressed"s,
           "frame_id"_a = "camera"s,
           "tensor_name"_a = ""s,
           "format"_a = "h264"s,
           "timestamp_metadata_keys"_a =
               "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns"s,
           "allocator"_a = std::shared_ptr<Allocator>(),
           "name"_a = "foxglove_compressed_video_adapter"s);

  py::class_<FoxglovePoseAdapterOp,
             PyFoxglovePoseAdapterOp,
             Operator,
             std::shared_ptr<FoxglovePoseAdapterOp>>(m,
                                                     "FoxglovePoseAdapterOp",
                                                     holoscan::doc::FoxglovePoseAdapterOp::doc)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<Allocator>,
                    const std::string&>(),
           "fragment"_a,
           "topic"_a = "/tf"s,
           "tensor_name"_a = ""s,
           "parent_frame_id"_a = "world"s,
           "child_frame_id"_a = "sensor"s,
           "format"_a = "matrix4x4"s,
           "timestamp_metadata_keys"_a =
               "acquisition_timestamp_ns,timestamp_ns,sensor_timestamp_ns"s,
           "allocator"_a = std::shared_ptr<Allocator>(),
           "name"_a = "foxglove_pose_adapter"s);

#if HOLOHUB_FOXGLOVE_HAS_EMITTER_RECEIVER_REGISTRY
  m.def("register_types", [](holoscan::EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<std::shared_ptr<FoxgloveBatch>>(
        "std::shared_ptr<holoscan::ops::FoxgloveBatch>"s);
    registry.add_emitter_receiver<std::vector<std::shared_ptr<FoxgloveBatch>>>(
        "std::vector<std::shared_ptr<holoscan::ops::FoxgloveBatch>>"s);
    registry.add_emitter_receiver<std::shared_ptr<FoxgloveImageAnnotations>>(
        "std::shared_ptr<holoscan::ops::FoxgloveImageAnnotations>"s);
    registry.add_emitter_receiver<std::vector<std::shared_ptr<FoxgloveImageAnnotations>>>(
        "std::vector<std::shared_ptr<holoscan::ops::FoxgloveImageAnnotations>>"s);
    registry.add_emitter_receiver<std::shared_ptr<FoxglovePointCloud>>(
        "std::shared_ptr<holoscan::ops::FoxglovePointCloud>"s);
    registry.add_emitter_receiver<std::vector<std::shared_ptr<FoxglovePointCloud>>>(
        "std::vector<std::shared_ptr<holoscan::ops::FoxglovePointCloud>>"s);
    registry.add_emitter_receiver<std::shared_ptr<FoxgloveKeyValue>>(
        "std::shared_ptr<holoscan::ops::FoxgloveKeyValue>"s);
    registry.add_emitter_receiver<std::vector<std::shared_ptr<FoxgloveKeyValue>>>(
        "std::vector<std::shared_ptr<holoscan::ops::FoxgloveKeyValue>>"s);
  });
#endif
}

}  // namespace holoscan::ops
