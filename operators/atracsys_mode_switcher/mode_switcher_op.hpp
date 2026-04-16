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

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "camera_calibration.hpp"
#include "hardware_mode_command.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "mode_switcher_keyboard.hpp"

namespace holoscan::ops {

enum class ReplayMode {
  kVisible = 1,
  kIr = 2,
  kStructured = 3,
  kTracking = 4,
};

class AtracsysModeSwitcherOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(AtracsysModeSwitcherOp, holoscan::Operator)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void setCameraCalibration(std::shared_ptr<CameraCalibration> camera_calibration);

 private:
  void handle_keyboard_request();
  ReplayMode requested_mode() const;
  void ensure_static_entities(const holoscan::ExecutionContext& context);
  void emit_cached_entity(const holoscan::gxf::Entity& entity, holoscan::OutputContext& op_output,
                          const char* port_name);
  void emit_placeholder_points(const holoscan::ExecutionContext& context,
                               holoscan::OutputContext& op_output);
  void emit_blank_base(const holoscan::ExecutionContext& context,
                       holoscan::OutputContext& op_output);
  atracsys::HardwareMode requested_hw_mode() const;

  static constexpr size_t kMaxFiducials = 6;
  static constexpr size_t kEntityRingSize = 4;

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> display_allocator_;
  holoscan::Parameter<bool> enable_keyboard_;
  holoscan::Parameter<int> initial_mode_;
  holoscan::Parameter<std::string> geometry_path_;

  ReplayMode mode_{ReplayMode::kVisible};
  ReplayMode last_base_mode_{ReplayMode::kVisible};
  atracsys::HardwareMode last_hw_mode_{atracsys::HardwareMode::kVisible};
  bool waiting_for_structured_frame_logged_{false};
  bool hw_command_pending_{true};
  std::shared_ptr<CameraCalibration> camera_calibration_;
  std::optional<holoscan::gxf::Entity> cached_visible_base_;
  std::optional<holoscan::gxf::Entity> cached_ir_base_;
  std::optional<holoscan::gxf::Entity> cached_structured_points_;
  std::optional<holoscan::gxf::Entity> cached_marker_poses_;
  std::vector<std::array<float, 3>> marker_local_geometry_mm_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> hidden_overlay_entities_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> hidden_marker_points_entities_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> placeholder_points_entities_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> blank_base_entities_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> mode_text_entities_;
  size_t static_entity_index_{0};
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> fiducial_text_coord_entities_;
  size_t fiducial_text_coords_entity_index_{0};
  std::array<std::array<float, 3>, kMaxFiducials> fiducial_text_coords_{};
  std::vector<float> marker_poses_host_;
  std::vector<std::array<float, 3>> transformed_marker_points_scratch_;
  std::vector<float> scene_marker_point_buffer_;
  std::vector<float> overlay_coords_scratch_;
  std::vector<std::array<float, 3>> overlay_label_points_scratch_;
  std::vector<holoscan::ops::HolovizOp::InputSpec> specs_;
  ModeSwitcherKeyboard keyboard_;
};

}  // namespace holoscan::ops
