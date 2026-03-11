/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <vector>
#include <deque>
#include <array>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

namespace holoscan::ops {

class VizOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VizOp)

  VizOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  // Helper methods
  std::shared_ptr<holoscan::Tensor> readTensorMap(const holoscan::TensorMap& tensormap, const std::string& tensor_name);
  void processIncomingTrajectory(const std::shared_ptr<holoscan::Tensor>& input_tensor);
  void processWindowTrajectory(const std::shared_ptr<holoscan::Tensor>& input_tensor);
  void computeTrajectoryBounds();
  void fitCameraToTrajectory();
  void handleCameraInput();
  void updateCamera();
  void renderTrajectory();
  void renderWindowTrajectory();
  void renderPointCloud(const std::shared_ptr<holoscan::Tensor>& input_tensor);
  void renderFramePoints(const std::shared_ptr<holoscan::Tensor>& frame, 
    const std::shared_ptr<holoscan::Tensor>& point_coords);
  void renderAxisGizmo();
  void renderGroundPlaneGrid();
  void renderCameraFrustum(const std::shared_ptr<holoscan::Tensor>& camera_position,
                           const std::shared_ptr<holoscan::Tensor>& camera_rotation);

  // Parameters
  holoscan::Parameter<holoscan::IOSpec*> in_;
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> device_allocator_;
  holoscan::Parameter<uint32_t> width_;
  holoscan::Parameter<uint32_t> height_;
  holoscan::Parameter<std::string> window_title_;
  holoscan::Parameter<bool> headless_;
  holoscan::CudaStreamHandler cuda_stream_handler_;
  holoscan::Parameter<bool> verbose_;
  
  // Full trajectory data (received each frame)
  std::vector<float> trajectory_host_;   // Host buffer for transformed trajectory
  float* dev_trajectory_ = nullptr;      // GPU buffer for trajectory
  size_t dev_trajectory_capacity_ = 0;   // Capacity of GPU buffer in floats
  size_t trajectory_num_points_ = 0;     // Current number of points in trajectory
  
  // Window trajectory (sliding window of last N points)
  std::deque<float> window_trajectory_buffer_;  // Stores x, y, z values sequentially
  float* dev_window_trajectory_ = nullptr;      // GPU buffer for window trajectory
  size_t dev_window_trajectory_capacity_ = 0;   // Capacity of GPU buffer in floats
  static constexpr size_t kWindowTrajectoryMaxPoints = 20;  // Max points in window trajectory
  
  // Point cloud (transformed with global rotation)
  float* dev_pointcloud_ = nullptr;             // GPU buffer for transformed point cloud
  size_t dev_pointcloud_capacity_ = 0;          // Capacity of GPU buffer in floats
  
  // Trajectory bounding box
  std::array<float, 3> bounds_min_ = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> bounds_max_ = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> bounds_center_ = {0.0f, 0.0f, 0.0f};
  float bounds_radius_ = 1.0f;  // Radius of bounding sphere
  
  // Orbit camera state
  float camera_distance_ = 2.0f;         // Distance from look-at point
  float camera_azimuth_ = 0.785f;        // Horizontal angle (radians), ~45 degrees
  float camera_elevation_ = 0.615f;      // Vertical angle (radians), ~35 degrees
  std::array<float, 3> camera_target_ = {0.0f, 0.0f, 0.0f};  // Look-at point
  
  // User zoom multiplier (1.0 = auto-fit, >1 = zoomed out, <1 = zoomed in)
  float user_zoom_factor_ = 1.0f;
  
  // Mouse state for camera control
  float last_mouse_x_ = 0.0f;
  float last_mouse_y_ = 0.0f;
  bool mouse_dragging_ = false;
  
  // Camera control sensitivity and limits
  static constexpr float kRotateSensitivity = 0.01f;
  static constexpr float kZoomSensitivity = 0.1f;
  static constexpr float kMinDistance = 0.1f;
  static constexpr float kMaxDistance = 1000.0f;
  static constexpr float kMinElevation = -1.5f;  // ~-86 degrees
  static constexpr float kMaxElevation = 1.5f;   // ~86 degrees
  static constexpr float kFitPadding = 1.5f;     // Padding factor for auto-fit
  static constexpr float kMinZoomFactor = 0.2f;
  static constexpr float kMaxZoomFactor = 5.0f;

  float left_side_ratio_ = 0.5f;
};

}  // namespace holoscan::ops
