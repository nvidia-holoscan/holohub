/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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

#include "mode_switcher_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::ops {

namespace {

constexpr const char* kStructuredPointsName = "structured_points";
constexpr const char* kBaseName = "base";
constexpr const char* kOverlayName = "fiducials_overlay";
constexpr const char* kMarkerPointsName = "marker_points";
constexpr const char* kMarkerPosesName = "marker_poses";
constexpr const char* kFiducialTextCoordsName = "fiducial_text_coords";
constexpr const char* kModeTextName = "mode_text";
constexpr const char* kModeTextPortName = "out_mode_text";

constexpr float kModeTextX = 0.01f;
constexpr float kModeTextY = 0.02f;
constexpr float kModeTextSize = 0.025f;
constexpr float kHiddenX = -10.0f;
constexpr float kHiddenY = -10.0f;
constexpr float kOverlayOvalSizeX = 0.012F;
constexpr float kOverlayOvalSizeY = 0.018F;

constexpr size_t kModeTextSpecIndex = 0;
constexpr size_t kFiducialTextSpecIndex = 1;

inline void check_cuda(cudaError_t code, const char* message) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(code));
  }
}

inline float max_abs_component(const std::array<float, 3>& values) {
  return std::max(std::max(std::fabs(values[0]), std::fabs(values[1])), std::fabs(values[2]));
}

inline std::string trim_copy(const std::string& value) {
  size_t first = 0;
  while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
    ++first;
  }

  size_t last = value.size();
  while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) { --last; }

  return value.substr(first, last - first);
}

inline std::optional<size_t> parse_fiducial_index(const std::string& section_name) {
  constexpr const char* kPrefix = "fiducial";
  constexpr size_t kPrefixLength = 8;
  if (section_name.rfind(kPrefix, 0) != 0 || section_name.size() <= kPrefixLength) {
    return std::nullopt;
  }

  size_t index = 0;
  for (size_t i = kPrefixLength; i < section_name.size(); ++i) {
    const char ch = section_name[i];
    if (ch < '0' || ch > '9') {
      return std::nullopt;
    }
    index = index * 10 + static_cast<size_t>(ch - '0');
  }
  return index;
}

inline bool pose_has_content(const float* pose) {
  for (size_t i = 0; i < 16; ++i) {
    if (std::fabs(pose[i]) > 1.0e-6F) {
      return true;
    }
  }
  return false;
}

struct PoseTransform {
  std::array<float, 16> row_major_mm{};
  bool used_row_major{false};
  bool source_was_meters{false};
};

inline std::vector<std::array<float, 3>> load_marker_geometry_points(
    const std::string& geometry_path) {
  std::ifstream input(geometry_path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open geometry file '" + geometry_path + "'");
  }

  struct GeometrySection {
    std::optional<size_t> index;
    std::optional<float> x;
    std::optional<float> y;
    std::optional<float> z;
  };

  std::vector<std::pair<size_t, std::array<float, 3>>> indexed_points;
  std::string current_section_name;
  GeometrySection current_section;
  size_t line_number = 0;

  auto flush_section = [&]() {
    if (!current_section.index.has_value()) {
      return;
    }
    if (!current_section.x.has_value() || !current_section.y.has_value() ||
        !current_section.z.has_value()) {
      throw std::runtime_error("Incomplete fiducial section [" + current_section_name + "] in '" +
                               geometry_path + "'");
    }
    indexed_points.emplace_back(
        current_section.index.value(),
        std::array<float, 3>{
            current_section.x.value(), current_section.y.value(), current_section.z.value()});
  };

  std::string line;
  while (std::getline(input, line)) {
    ++line_number;
    const std::string trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed[0] == ';' || trimmed[0] == '#') {
      continue;
    }

    if (trimmed.front() == '[' && trimmed.back() == ']') {
      flush_section();
      current_section_name = trim_copy(trimmed.substr(1, trimmed.size() - 2));
      current_section = {};
      current_section.index = parse_fiducial_index(current_section_name);
      continue;
    }

    if (!current_section.index.has_value()) {
      continue;
    }

    const size_t separator = trimmed.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("Malformed geometry line " + std::to_string(line_number) + " in '" +
                               geometry_path + "'");
    }

    const std::string key = trim_copy(trimmed.substr(0, separator));
    const std::string value = trim_copy(trimmed.substr(separator + 1));

    float numeric_value = 0.0F;
    try {
      numeric_value = std::stof(value);
    } catch (const std::exception&) {
      throw std::runtime_error("Invalid numeric value '" + value + "' in '" + geometry_path + "'");
    }

    if (key == "x") {
      current_section.x = numeric_value;
    } else if (key == "y") {
      current_section.y = numeric_value;
    } else if (key == "z") {
      current_section.z = numeric_value;
    }
  }

  flush_section();
  if (indexed_points.empty()) {
    throw std::runtime_error("No fiducial points found in geometry file '" + geometry_path + "'");
  }

  std::sort(indexed_points.begin(), indexed_points.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  std::vector<std::array<float, 3>> local_points;
  local_points.reserve(indexed_points.size());
  for (const auto& entry : indexed_points) { local_points.push_back(entry.second); }
  return local_points;
}

inline PoseTransform decode_pose_transform(const float* pose) {
  const std::array<float, 3> row_major_translation{pose[3], pose[7], pose[11]};
  const std::array<float, 3> column_major_translation{pose[12], pose[13], pose[14]};

  const float row_major_mag = max_abs_component(row_major_translation);
  const float column_major_mag = max_abs_component(column_major_translation);

  PoseTransform transform;
  if (row_major_mag > 1.0e-6F && column_major_mag <= 1.0e-6F) {
    transform.used_row_major = true;
  } else if (row_major_mag > column_major_mag) {
    transform.used_row_major = true;
  }

  if (transform.used_row_major) {
    std::copy(pose, pose + 16, transform.row_major_mm.begin());
  } else {
    for (int row = 0; row < 4; ++row) {
      for (int column = 0; column < 4; ++column) {
        transform.row_major_mm[row * 4 + column] = pose[column * 4 + row];
      }
    }
  }

  std::array<float, 3> translation_mm{
      transform.row_major_mm[3], transform.row_major_mm[7], transform.row_major_mm[11]};
  const float chosen_mag = max_abs_component(translation_mm);
  if (chosen_mag > 1.0e-6F && chosen_mag < 10.0F) {
    transform.row_major_mm[3] *= 1000.0F;
    transform.row_major_mm[7] *= 1000.0F;
    transform.row_major_mm[11] *= 1000.0F;
    transform.source_was_meters = true;
  }

  return transform;
}

inline std::array<float, 3> transform_local_geometry_point(
    const PoseTransform& pose_transform, const std::array<float, 3>& local_point) {
  return {pose_transform.row_major_mm[0] * local_point[0] +
              pose_transform.row_major_mm[1] * local_point[1] +
              pose_transform.row_major_mm[2] * local_point[2] + pose_transform.row_major_mm[3],
          pose_transform.row_major_mm[4] * local_point[0] +
              pose_transform.row_major_mm[5] * local_point[1] +
              pose_transform.row_major_mm[6] * local_point[2] + pose_transform.row_major_mm[7],
          pose_transform.row_major_mm[8] * local_point[0] +
              pose_transform.row_major_mm[9] * local_point[1] +
              pose_transform.row_major_mm[10] * local_point[2] + pose_transform.row_major_mm[11]};
}

inline std::optional<std::array<float, 2>> project_marker_to_overlay(
    const std::array<float, 3>& point_mm, const CameraCalibration& calibration) {
  constexpr float kMinDepthMm = 1.0e-3F;
  const float z = point_mm[2];
  if (!std::isfinite(point_mm[0]) || !std::isfinite(point_mm[1]) || !std::isfinite(z) ||
      z <= kMinDepthMm || calibration.image_width == 0 || calibration.image_height == 0) {
    return std::nullopt;
  }

  const float x = point_mm[0] / z;
  const float y = point_mm[1] / z;
  const float r2 = x * x + y * y;
  const float r4 = r2 * r2;
  const float r6 = r4 * r2;
  const float radial = 1.0F + calibration.distortion[0] * r2 + calibration.distortion[1] * r4 +
                       calibration.distortion[4] * r6;
  const float x_tangential =
      2.0F * calibration.distortion[2] * x * y + calibration.distortion[3] * (r2 + 2.0F * x * x);
  const float y_tangential =
      calibration.distortion[2] * (r2 + 2.0F * y * y) + 2.0F * calibration.distortion[3] * x * y;
  const float x_distorted = x * radial + x_tangential;
  const float y_distorted = y * radial + y_tangential;

  const float u = calibration.fx * x_distorted + calibration.skew * y_distorted + calibration.cx;
  const float v = calibration.fy * y_distorted + calibration.cy;
  if (!std::isfinite(u) || !std::isfinite(v)) {
    return std::nullopt;
  }

  return std::array<float, 2>{u / static_cast<float>(calibration.image_width),
                              v / static_cast<float>(calibration.image_height)};
}

inline const char* mode_label_text(ReplayMode mode, ReplayMode last_base_mode) {
  switch (mode) {
    case ReplayMode::kVisible:
      return "Visible Mode";
    case ReplayMode::kIr:
      return "Infrared Mode";
    case ReplayMode::kStructured:
      return "Structured Light Mode";
    case ReplayMode::kTracking:
      switch (last_base_mode) {
        case ReplayMode::kVisible:
          return "Visible Mode + Tracking";
        case ReplayMode::kIr:
          return "Infrared Mode + Tracking";
        case ReplayMode::kStructured:
          return "Structured Light Mode + Tracking";
        default:
          return "Mode + Tracking";
      }
    default:
      return "Visible Mode";
  }
}

template <typename T>
void add_device_tensor_to_entity(nvidia::gxf::Entity& entity,
                                 const holoscan::ExecutionContext& context,
                                 const std::shared_ptr<holoscan::Allocator>& allocator,
                                 const char* tensor_name, const nvidia::gxf::Shape& shape,
                                 const T* data, size_t element_count, const char* tensor_error,
                                 const char* copy_error) {
  auto tensor = entity.add<nvidia::gxf::Tensor>(tensor_name);
  if (!tensor) {
    throw std::runtime_error(tensor_error);
  }

  auto alloc =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator->gxf_cid());
  tensor.value()->reshape<T>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc.value());
  check_cuda(
      cudaMemcpy(
          tensor.value()->pointer(), data, element_count * sizeof(T), cudaMemcpyHostToDevice),
      copy_error);
}

template <typename T>
holoscan::gxf::Entity create_device_tensor_entity(
    const holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Allocator>& allocator, const char* tensor_name,
    const nvidia::gxf::Shape& shape, const T* data, size_t element_count, const char* entity_error,
    const char* tensor_error, const char* copy_error) {
  auto msg = nvidia::gxf::Entity::New(context.context());
  if (!msg) {
    throw std::runtime_error(entity_error);
  }

  add_device_tensor_to_entity<T>(msg.value(),
                                 context,
                                 allocator,
                                 tensor_name,
                                 shape,
                                 data,
                                 element_count,
                                 tensor_error,
                                 copy_error);

  return holoscan::gxf::Entity(std::move(msg.value()));
}

inline holoscan::gxf::Entity create_overlay_entity(
    const holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Allocator>& allocator,
    const std::vector<float>& overlay_coords) {
  auto msg = nvidia::gxf::Entity::New(context.context());
  if (!msg) {
    throw std::runtime_error("Failed to allocate marker overlay message");
  }

  add_device_tensor_to_entity<float>(
      msg.value(),
      context,
      allocator,
      kOverlayName,
      nvidia::gxf::Shape{static_cast<int32_t>(overlay_coords.size() / 4), 4},
      overlay_coords.data(),
      overlay_coords.size(),
      "Failed to add marker overlay tensor",
      "Failed to upload marker overlay coordinates");
  return holoscan::gxf::Entity(std::move(msg.value()));
}

inline holoscan::gxf::Entity create_marker_points_entity(
    const holoscan::ExecutionContext& context,
    const std::shared_ptr<holoscan::Allocator>& allocator, const float* points,
    size_t point_count) {
  auto msg = nvidia::gxf::Entity::New(context.context());
  if (!msg) {
    throw std::runtime_error("Failed to allocate marker points message");
  }

  const nvidia::gxf::Shape shape{static_cast<int32_t>(point_count / 3), 3};
  add_device_tensor_to_entity<float>(msg.value(),
                                     context,
                                     allocator,
                                     kMarkerPointsName,
                                     shape,
                                     points,
                                     point_count,
                                     "Failed to add marker points tensor",
                                     "Failed to upload marker point coordinates");
  return holoscan::gxf::Entity(std::move(msg.value()));
}

void upload_to_tensor(const holoscan::gxf::Entity& entity, const char* tensor_name,
                      const void* data, size_t bytes, const char* tensor_error,
                      const char* copy_error) {
  auto gxf_entity = nvidia::gxf::Entity(entity);
  auto tensor = gxf_entity.get<nvidia::gxf::Tensor>(tensor_name);
  if (!tensor) {
    throw std::runtime_error(tensor_error);
  }
  check_cuda(cudaMemcpy(tensor.value()->pointer(), data, bytes, cudaMemcpyHostToDevice),
             copy_error);
}

}  // namespace

void AtracsysModeSwitcherOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("in_visible_base").condition(holoscan::ConditionType::kNone);
  spec.input<holoscan::gxf::Entity>("in_ir_base").condition(holoscan::ConditionType::kNone);
  spec.input<holoscan::gxf::Entity>("in_structured_points")
      .condition(holoscan::ConditionType::kNone);
  spec.input<holoscan::gxf::Entity>("in_marker_poses").condition(holoscan::ConditionType::kNone);

  spec.output<holoscan::gxf::Entity>("out_base");
  spec.output<holoscan::gxf::Entity>("out_overlay");
  spec.output<holoscan::gxf::Entity>("out_marker_points");
  spec.output<holoscan::gxf::Entity>("out_points");
  spec.output<holoscan::gxf::Entity>(kModeTextPortName);
  spec.output<holoscan::gxf::Entity>("out_fiducial_text_coords");
  spec.output<std::vector<holoscan::ops::HolovizOp::InputSpec>>("out_specs");
  spec.output<std::shared_ptr<atracsys::HardwareModeCommand>>("out_hw_cmd")
      .condition(holoscan::ConditionType::kNone);

  spec.param(display_allocator_,
             "display_allocator",
             "DisplayAllocator",
             "Output allocator for display-facing placeholder and text tensors");
  spec.param(enable_keyboard_,
             "enable_keyboard",
             "EnableKeyboard",
             "Allow stdin keyboard toggles 1=visible,2=ir,3=structured(points),4=tracking",
             true);
  spec.param(
      initial_mode_, "initial_mode", "InitialMode", "Start mode 1=Visible,2=IR,3=SL,4=Tracking", 1);
  spec.param(geometry_path_,
             "geometry_path",
             "GeometryPath",
             "Path to the rigid-body geometry .ini file used for marker rendering",
             std::string("geometry10.ini"));
}

void AtracsysModeSwitcherOp::setCameraCalibration(
    std::shared_ptr<CameraCalibration> camera_calibration) {
  camera_calibration_ = std::move(camera_calibration);
}

void AtracsysModeSwitcherOp::start() {
  mode_ = requested_mode();
  if (mode_ != ReplayMode::kTracking) {
    last_base_mode_ = mode_;
  }
  last_hw_mode_ = requested_hw_mode();
  hw_command_pending_ = true;

  waiting_for_structured_frame_logged_ = false;
  cached_visible_base_.reset();
  cached_ir_base_.reset();
  cached_structured_points_.reset();
  cached_marker_poses_.reset();
  marker_local_geometry_mm_ = load_marker_geometry_points(geometry_path_.get());
  for (auto& entity : hidden_overlay_entities_) { entity.reset(); }
  for (auto& entity : hidden_marker_points_entities_) { entity.reset(); }
  for (auto& entity : placeholder_points_entities_) { entity.reset(); }
  for (auto& entity : blank_base_entities_) { entity.reset(); }
  for (auto& entity : mode_text_entities_) { entity.reset(); }
  static_entity_index_ = 0;
  for (auto& entity : fiducial_text_coord_entities_) { entity.reset(); }
  fiducial_text_coords_entity_index_ = 0;
  marker_poses_host_.clear();
  transformed_marker_points_scratch_.clear();
  scene_marker_point_buffer_.clear();
  overlay_coords_scratch_.clear();
  overlay_label_points_scratch_.clear();
  specs_.clear();
  specs_.reserve(2);

  if (!camera_calibration_ || !camera_calibration_->valid()) {
    throw std::runtime_error(
        "AtracsysModeSwitcherOp: app must provide a valid camera calibration before start()");
  }

  auto make_spec = [](const char* tensor_name, float r, float g, float b, float a) {
    holoscan::ops::HolovizOp::InputSpec spec;
    spec.tensor_name_ = tensor_name;
    spec.type_ = holoscan::ops::HolovizOp::InputType::TEXT;
    spec.color_ = {r, g, b, a};
    return spec;
  };

  specs_.push_back(make_spec(kModeTextName, 1.0f, 1.0f, 1.0f, 1.0f));
  specs_.back().text_.push_back(mode_label_text(mode_, last_base_mode_));
  specs_.push_back(make_spec(kFiducialTextCoordsName, 0.95f, 0.95f, 0.95f, 1.0f));
  specs_.back().text_.resize(kMaxFiducials);
  for (auto& text : specs_.back().text_) { text.reserve(32); }

  HOLOSCAN_LOG_INFO(
      "Loaded {} local fiducials from {}", marker_local_geometry_mm_.size(), geometry_path_.get());
  HOLOSCAN_LOG_INFO("Mode switcher started in mode {}", static_cast<int>(mode_));
  keyboard_.start(enable_keyboard_.get());
}

void AtracsysModeSwitcherOp::stop() {
  keyboard_.stop();
  cached_visible_base_.reset();
  cached_ir_base_.reset();
  cached_structured_points_.reset();
  cached_marker_poses_.reset();
  marker_local_geometry_mm_.clear();
  for (auto& entity : hidden_overlay_entities_) { entity.reset(); }
  for (auto& entity : hidden_marker_points_entities_) { entity.reset(); }
  for (auto& entity : placeholder_points_entities_) { entity.reset(); }
  for (auto& entity : blank_base_entities_) { entity.reset(); }
  for (auto& entity : mode_text_entities_) { entity.reset(); }
  static_entity_index_ = 0;
  for (auto& entity : fiducial_text_coord_entities_) { entity.reset(); }
  fiducial_text_coords_entity_index_ = 0;
  marker_poses_host_.clear();
  transformed_marker_points_scratch_.clear();
  scene_marker_point_buffer_.clear();
  overlay_coords_scratch_.clear();
  overlay_label_points_scratch_.clear();
  specs_.clear();
  holoscan::Operator::stop();
}

void AtracsysModeSwitcherOp::ensure_static_entities(const holoscan::ExecutionContext& context) {
  if (hidden_overlay_entities_[0] && hidden_marker_points_entities_[0] &&
      placeholder_points_entities_[0] && blank_base_entities_[0] && mode_text_entities_[0] &&
      fiducial_text_coord_entities_[0]) {
    return;
  }

  static const std::array<float, 4> kHiddenOverlayCoords{
      -1.0F, -1.0F, kOverlayOvalSizeX, kOverlayOvalSizeY};
  static const std::array<float, 3> kPlaceholderPoint{1.0e9F, 1.0e9F, 1.0e9F};
  static const std::array<uint8_t, 4> kBlankBasePixels{0, 0, 0, 0};
  std::array<std::array<float, 3>, kMaxFiducials> hidden_fiducial_coords{};
  for (auto& coord : hidden_fiducial_coords) { coord = {kHiddenX, kHiddenY, 0.04F}; }

  for (auto& entity : hidden_overlay_entities_) {
    entity = create_device_tensor_entity<float>(context,
                                                display_allocator_.get(),
                                                kOverlayName,
                                                nvidia::gxf::Shape{1, 4},
                                                kHiddenOverlayCoords.data(),
                                                kHiddenOverlayCoords.size(),
                                                "Failed to allocate hidden overlay message",
                                                "Failed to add hidden overlay tensor",
                                                "Failed to upload hidden overlay coordinates");
  }

  for (auto& entity : blank_base_entities_) {
    entity = create_device_tensor_entity<uint8_t>(context,
                                                  display_allocator_.get(),
                                                  kBaseName,
                                                  nvidia::gxf::Shape{1, 1, 4},
                                                  kBlankBasePixels.data(),
                                                  kBlankBasePixels.size(),
                                                  "Failed to allocate blank base message",
                                                  "Failed to add blank base tensor",
                                                  "Failed to upload blank base tensor");
  }

  for (auto& entity : placeholder_points_entities_) {
    entity = create_device_tensor_entity<float>(context,
                                                display_allocator_.get(),
                                                kStructuredPointsName,
                                                nvidia::gxf::Shape{1, 3},
                                                kPlaceholderPoint.data(),
                                                kPlaceholderPoint.size(),
                                                "Failed to allocate placeholder points message",
                                                "Failed to add placeholder points tensor",
                                                "Failed to upload placeholder point coords");
  }

  for (auto& entity : hidden_marker_points_entities_) {
    entity = create_marker_points_entity(
        context, display_allocator_.get(), kPlaceholderPoint.data(), kPlaceholderPoint.size());
  }
  const std::array<float, 3> mode_text_coords{kModeTextX, kModeTextY, kModeTextSize};
  for (auto& entity : mode_text_entities_) {
    entity = create_device_tensor_entity<float>(context,
                                                display_allocator_.get(),
                                                kModeTextName,
                                                nvidia::gxf::Shape{1, 3},
                                                mode_text_coords.data(),
                                                mode_text_coords.size(),
                                                "Failed to allocate mode text message",
                                                "Failed to add mode text tensor",
                                                "Failed to upload mode text coords");
  }

  for (auto& entity : fiducial_text_coord_entities_) {
    entity = create_device_tensor_entity<float>(
        context,
        display_allocator_.get(),
        kFiducialTextCoordsName,
        nvidia::gxf::Shape{static_cast<int32_t>(kMaxFiducials), 3},
        reinterpret_cast<const float*>(hidden_fiducial_coords.data()),
        kMaxFiducials * 3,
        "Failed to allocate fiducial text coords message",
        "Failed to add fiducial text coords tensor",
        "Failed to upload fiducial text coordinates");
  }
}

void AtracsysModeSwitcherOp::emit_cached_entity(const holoscan::gxf::Entity& entity,
                                                holoscan::OutputContext& op_output,
                                                const char* port_name) {
  holoscan::gxf::Entity out(entity);
  op_output.emit(out, port_name);
}

ReplayMode AtracsysModeSwitcherOp::requested_mode() const {
  switch (initial_mode_.get()) {
    case 1:
      return ReplayMode::kVisible;
    case 2:
      return ReplayMode::kIr;
    case 3:
      return ReplayMode::kStructured;
    case 4:
      return ReplayMode::kTracking;
    default:
      return ReplayMode::kVisible;
  }
}

atracsys::HardwareMode AtracsysModeSwitcherOp::requested_hw_mode() const {
  switch (mode_) {
    case ReplayMode::kVisible:
      return atracsys::HardwareMode::kVisible;
    case ReplayMode::kIr:
    case ReplayMode::kTracking:
      return atracsys::HardwareMode::kInfrared;
    case ReplayMode::kStructured:
      return atracsys::HardwareMode::kStructured;
    default:
      return atracsys::HardwareMode::kVisible;
  }
}

void AtracsysModeSwitcherOp::handle_keyboard_request() {
  const auto key = keyboard_.poll_key();
  if (!key.has_value()) {
    return;
  }

  ReplayMode next = mode_;
  switch (key.value()) {
    case '1':
      next = ReplayMode::kVisible;
      break;
    case '2':
      next = ReplayMode::kIr;
      break;
    case '3':
      next = ReplayMode::kStructured;
      break;
    case '4':
      next = ReplayMode::kTracking;
      break;
    default:
      break;
  }

  if (next != mode_) {
    if (next == ReplayMode::kTracking) {
      last_base_mode_ = mode_;
    } else {
      last_base_mode_ = next;
    }
    mode_ = next;
    waiting_for_structured_frame_logged_ = false;
    const auto next_hw_mode = requested_hw_mode();
    if (next_hw_mode != last_hw_mode_) {
      last_hw_mode_ = next_hw_mode;
      hw_command_pending_ = true;
    }

    HOLOSCAN_LOG_INFO("Mode switcher switched to mode {}", static_cast<int>(mode_));
  }
}

void AtracsysModeSwitcherOp::emit_placeholder_points(const holoscan::ExecutionContext& context,
                                                     holoscan::OutputContext& op_output) {
  ensure_static_entities(context);
  emit_cached_entity(
      placeholder_points_entities_[static_entity_index_].value(), op_output, "out_points");
}

void AtracsysModeSwitcherOp::emit_blank_base(const holoscan::ExecutionContext& context,
                                             holoscan::OutputContext& op_output) {
  ensure_static_entities(context);
  emit_cached_entity(blank_base_entities_[static_entity_index_].value(), op_output, "out_base");
}

void AtracsysModeSwitcherOp::compute(holoscan::InputContext& op_input,
                                     holoscan::OutputContext& op_output,
                                     holoscan::ExecutionContext& context) {
  ensure_static_entities(context);
  handle_keyboard_request();

  if (hw_command_pending_) {
    auto cmd = std::make_shared<atracsys::HardwareModeCommand>();
    cmd->mode = last_hw_mode_;
    op_output.emit(cmd, "out_hw_cmd");
    hw_command_pending_ = false;
  }

  auto visible = op_input.receive<holoscan::gxf::Entity>("in_visible_base");
  auto ir = op_input.receive<holoscan::gxf::Entity>("in_ir_base");
  auto structured_points = op_input.receive<holoscan::gxf::Entity>("in_structured_points");
  auto marker_poses = op_input.receive<holoscan::gxf::Entity>("in_marker_poses");

  if (visible && !visible.value().is_null()) {
    cached_visible_base_.emplace(std::move(visible.value()));
  }
  if (ir && !ir.value().is_null()) {
    cached_ir_base_.emplace(std::move(ir.value()));
  }

  bool emitted_base = false;
  switch (mode_) {
    case ReplayMode::kVisible:
      if (cached_visible_base_.has_value() && !cached_visible_base_.value().is_null()) {
        holoscan::gxf::Entity selected_base(cached_visible_base_.value());
        op_output.emit(selected_base, "out_base");
        emitted_base = true;
      }
      break;
    case ReplayMode::kIr:
      if (cached_ir_base_.has_value() && !cached_ir_base_.value().is_null()) {
        holoscan::gxf::Entity selected_base(cached_ir_base_.value());
        op_output.emit(selected_base, "out_base");
        emitted_base = true;
      }
      break;
    case ReplayMode::kStructured:
      break;
    case ReplayMode::kTracking:
      switch (last_base_mode_) {
        case ReplayMode::kIr:
          if (cached_ir_base_.has_value() && !cached_ir_base_.value().is_null()) {
            holoscan::gxf::Entity selected_base(cached_ir_base_.value());
            op_output.emit(selected_base, "out_base");
            emitted_base = true;
          }
          break;
        case ReplayMode::kVisible:
          if (cached_visible_base_.has_value() && !cached_visible_base_.value().is_null()) {
            holoscan::gxf::Entity selected_base(cached_visible_base_.value());
            op_output.emit(selected_base, "out_base");
            emitted_base = true;
          }
          break;
        case ReplayMode::kStructured:
        case ReplayMode::kTracking:
          break;
      }
      break;
  }
  if (!emitted_base) {
    emit_blank_base(context, op_output);
  }

  const bool had_cached_structured = cached_structured_points_.has_value();
  if (structured_points && !structured_points.value().is_null()) {
    cached_structured_points_.emplace(std::move(structured_points.value()));
    if (mode_ == ReplayMode::kStructured) {
      waiting_for_structured_frame_logged_ = false;
      if (!had_cached_structured) {
        HOLOSCAN_LOG_INFO("Mode switcher received first structured_points frame");
      }
    }
  }

  const bool structured_tracking_active =
      (mode_ == ReplayMode::kTracking && last_base_mode_ == ReplayMode::kStructured);
  const bool structured_active = (mode_ == ReplayMode::kStructured) || structured_tracking_active;
  if (structured_active) {
    if (cached_structured_points_.has_value() && !cached_structured_points_.value().is_null()) {
      holoscan::gxf::Entity out_points(cached_structured_points_.value());
      op_output.emit(out_points, "out_points");
    } else {
      if (!waiting_for_structured_frame_logged_) {
        HOLOSCAN_LOG_WARN("Structured output active, waiting for first structured_points frame");
        waiting_for_structured_frame_logged_ = true;
      }
      emit_placeholder_points(context, op_output);
    }
  } else {
    emit_placeholder_points(context, op_output);
    waiting_for_structured_frame_logged_ = false;
  }

  if (marker_poses && !marker_poses.value().is_null()) {
    cached_marker_poses_.emplace(std::move(marker_poses.value()));
  }

  transformed_marker_points_scratch_.clear();
  scene_marker_point_buffer_.clear();
  overlay_coords_scratch_.clear();
  overlay_label_points_scratch_.clear();
  if (cached_marker_poses_.has_value() && !cached_marker_poses_.value().is_null()) {
    auto gxf_entity = nvidia::gxf::Entity(cached_marker_poses_.value());
    auto poses_tensor = gxf_entity.get<nvidia::gxf::Tensor>(kMarkerPosesName);
    if (poses_tensor) {
      const uint32_t num_markers = poses_tensor.value()->shape().dimension(0);
      transformed_marker_points_scratch_.reserve(marker_local_geometry_mm_.size());
      scene_marker_point_buffer_.reserve(marker_local_geometry_mm_.size() * 3);

      marker_poses_host_.resize(num_markers * 16);
      const auto storage = poses_tensor.value()->storage_type();
      if (storage == nvidia::gxf::MemoryStorageType::kHost ||
          storage == nvidia::gxf::MemoryStorageType::kSystem) {
        std::memcpy(marker_poses_host_.data(),
                    poses_tensor.value()->pointer(),
                    num_markers * 16 * sizeof(float));
      } else {
        check_cuda(cudaMemcpy(marker_poses_host_.data(),
                              poses_tensor.value()->pointer(),
                              num_markers * 16 * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "Failed to copy marker poses to host for point extraction");
      }

      for (uint32_t i = 0; i < num_markers; ++i) {
        const float* pose = marker_poses_host_.data() + i * 16;
        if (!pose_has_content(pose)) {
          continue;
        }

        const auto pose_transform = decode_pose_transform(pose);
        for (const auto& local_point : marker_local_geometry_mm_) {
          transformed_marker_points_scratch_.push_back(
              transform_local_geometry_point(pose_transform, local_point));
        }
        break;
      }

      for (const auto& world_point : transformed_marker_points_scratch_) {
        scene_marker_point_buffer_.push_back(world_point[0]);
        scene_marker_point_buffer_.push_back(world_point[1]);
        scene_marker_point_buffer_.push_back(world_point[2]);
      }
    }
  }

  const bool overlay_active =
      (mode_ == ReplayMode::kTracking) &&
      (last_base_mode_ == ReplayMode::kVisible || last_base_mode_ == ReplayMode::kIr);
  const bool marker_points_active = structured_tracking_active;
  const bool live_projected_overlay_active =
      overlay_active && camera_calibration_ && camera_calibration_->valid();

  auto& overlay_coords = overlay_coords_scratch_;
  auto& overlay_label_points = overlay_label_points_scratch_;
  bool emitted_overlay = false;
  if (live_projected_overlay_active && !transformed_marker_points_scratch_.empty()) {
    overlay_coords.reserve(transformed_marker_points_scratch_.size() * 4);
    overlay_label_points.reserve(
        std::min(transformed_marker_points_scratch_.size(), kMaxFiducials));
    for (const auto& transformed_point : transformed_marker_points_scratch_) {
      const auto overlay_point = project_marker_to_overlay(transformed_point, *camera_calibration_);
      if (!overlay_point.has_value()) {
        continue;
      }

      overlay_coords.push_back(overlay_point.value()[0]);
      overlay_coords.push_back(overlay_point.value()[1]);
      overlay_coords.push_back(kOverlayOvalSizeX);
      overlay_coords.push_back(kOverlayOvalSizeY);
      if (overlay_label_points.size() < kMaxFiducials) {
        overlay_label_points.push_back(transformed_point);
      }
    }

    if (!overlay_coords.empty()) {
      auto projected_overlay =
          create_overlay_entity(context, display_allocator_.get(), overlay_coords);
      op_output.emit(projected_overlay, "out_overlay");
      emitted_overlay = true;
    }
  }
  if (!emitted_overlay) {
    emit_cached_entity(
        hidden_overlay_entities_[static_entity_index_].value(), op_output, "out_overlay");
  }

  bool emitted_marker_points = false;
  if (marker_points_active && !scene_marker_point_buffer_.empty()) {
    auto marker_points_entity = create_marker_points_entity(context,
                                                            display_allocator_.get(),
                                                            scene_marker_point_buffer_.data(),
                                                            scene_marker_point_buffer_.size());
    op_output.emit(marker_points_entity, "out_marker_points");
    emitted_marker_points = true;
  }
  if (!emitted_marker_points) {
    emit_cached_entity(hidden_marker_points_entities_[static_entity_index_].value(),
                       op_output,
                       "out_marker_points");
  }

  specs_[kModeTextSpecIndex].text_[0] = mode_label_text(mode_, last_base_mode_);
  emit_cached_entity(
      mode_text_entities_[static_entity_index_].value(), op_output, kModeTextPortName);

  for (size_t i = 0; i < kMaxFiducials; ++i) {
    fiducial_text_coords_[i] = {kHiddenX, kHiddenY, 0.04F};
    specs_[kFiducialTextSpecIndex].text_[i].clear();
  }

  if (overlay_active && !overlay_coords.empty()) {
    for (size_t i = 0; i < overlay_coords.size() / 4 && i < kMaxFiducials; ++i) {
      const float x = overlay_coords[i * 4 + 0];
      const float y = overlay_coords[i * 4 + 1];
      if (x < 0.0F || x > 1.0F || y < 0.0F || y > 1.0F) {
        continue;
      }

      fiducial_text_coords_[i] = {x, y - 0.03F, 0.028F};
      char buffer[48];
      const auto& world = overlay_label_points[i];
      std::snprintf(buffer, sizeof(buffer), "X:%.0f Y:%.0f Z:%.0f", world[0], world[1], world[2]);
      specs_[kFiducialTextSpecIndex].text_[i] = buffer;
    }
  }

  const auto& fiducial_entity =
      fiducial_text_coord_entities_[fiducial_text_coords_entity_index_].value();
  upload_to_tensor(fiducial_entity,
                   kFiducialTextCoordsName,
                   reinterpret_cast<const float*>(fiducial_text_coords_.data()),
                   kMaxFiducials * 3 * sizeof(float),
                   "Failed to retrieve fiducial text coords tensor",
                   "Failed to upload fiducial text coords");
  emit_cached_entity(fiducial_entity, op_output, "out_fiducial_text_coords");
  fiducial_text_coords_entity_index_ =
      (fiducial_text_coords_entity_index_ + 1) % fiducial_text_coord_entities_.size();
  static_entity_index_ = (static_entity_index_ + 1) % kEntityRingSize;

  op_output.emit(specs_, "out_specs");
}

}  // namespace holoscan::ops
