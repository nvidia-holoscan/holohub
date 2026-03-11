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

#include "viz.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/imgui.h>

#include <gxf/std/tensor.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#define CUDA_TRY(stmt)                                                                            \
  {                                                                                               \
    cudaError_t cuda_error = stmt;                                                                \
    if (cuda_error != cudaSuccess) {                                                              \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(cuda_error));     \
    }                                                                                             \
  }

namespace viz = holoscan::viz;

namespace holoscan::ops {

void VizOp::setup(OperatorSpec& spec) {
  auto& in_tensor = spec.input<gxf::Entity>("receivers");
  spec.param(in_, "receivers", "Input", "Input port.", &in_tensor);

  spec.param(device_allocator_, "device_allocator", "Allocator", "Output Allocator");

  spec.param(width_, "width", "Width", "Window width", 1280u);
  spec.param(height_, "height", "Height", "Window height", 720u);
  spec.param(window_title_, "window_title", "Window Title", "Window title", std::string("Holoviz"));
  spec.param(headless_, "headless", "Headless", "Headless mode", false);
  spec.param(verbose_, "verbose", "Verbose", "Verbose mode", false);

  cuda_stream_handler_.define_params(spec);
}

void VizOp::start() {
  viz::Init(width_.get(), height_.get(), window_title_.get().c_str(),
            headless_.get() ? viz::InitFlags::HEADLESS : viz::InitFlags::NONE);
  
  // GPU buffers will be allocated dynamically as needed
  dev_trajectory_ = nullptr;
  dev_trajectory_capacity_ = 0;
  
  // Pre-allocate window trajectory GPU buffer (fixed size)
  dev_window_trajectory_capacity_ = kWindowTrajectoryMaxPoints * 3;
  CUDA_TRY(cudaMalloc(&dev_window_trajectory_, dev_window_trajectory_capacity_ * sizeof(float)));

  left_side_ratio_ = 480.0f / width_.get();
}

void VizOp::stop() {
  if (dev_trajectory_) {
    cudaFree(dev_trajectory_);
    dev_trajectory_ = nullptr;
  }
  if (dev_window_trajectory_) {
    cudaFree(dev_window_trajectory_);
    dev_window_trajectory_ = nullptr;
  }
  if (dev_pointcloud_) {
    cudaFree(dev_pointcloud_);
    dev_pointcloud_ = nullptr;
  }
  viz::Shutdown();
}

void VizOp::processIncomingTrajectory(const std::shared_ptr<holoscan::Tensor>& input_tensor) {
  // The full trajectory is received each frame - copy and transform it
  auto num_points = input_tensor->shape()[0];
  size_t num_floats = num_points * 3;  // 3 floats per point (x, y, z)
  
  // Resize host buffer if needed
  trajectory_host_.resize(num_floats);
  trajectory_num_points_ = num_points;
  
  // Copy from GPU to host
  CUDA_TRY(cudaMemcpy(trajectory_host_.data(), input_tensor->data(), 
                       num_floats * sizeof(float), cudaMemcpyDeviceToHost));
  
  // Apply global rotation to flip Y and Z: [[1,0,0],[0,-1,0],[0,0,-1]]
  // Transform each point: (x, y, z) -> (x, -y, -z)
  for (size_t i = 0; i < num_points; ++i) {
    trajectory_host_[i * 3 + 1] = -trajectory_host_[i * 3 + 1];  // flip Y
    trajectory_host_[i * 3 + 2] = -trajectory_host_[i * 3 + 2];  // flip Z
  }
  
  // Reallocate GPU buffer if needed
  if (num_floats > dev_trajectory_capacity_) {
    if (dev_trajectory_) {
      cudaFree(dev_trajectory_);
    }
    // Allocate with some extra capacity to reduce reallocations
    dev_trajectory_capacity_ = num_floats * 2;
    CUDA_TRY(cudaMalloc(&dev_trajectory_, dev_trajectory_capacity_ * sizeof(float)));
  }
  
  // Copy transformed trajectory to GPU
  if (num_floats >= 6) {  // Need at least 2 points for a line
    CUDA_TRY(cudaMemcpy(dev_trajectory_, trajectory_host_.data(), 
                         num_floats * sizeof(float), cudaMemcpyHostToDevice));
  }
}

void VizOp::processWindowTrajectory(const std::shared_ptr<holoscan::Tensor>& input_tensor) {
  // Window trajectory receives one point at a time - accumulate in sliding window
  auto num_points = input_tensor->shape()[0];
  size_t num_floats = num_points * 3;
  
  std::vector<float> host_points(num_floats);
  CUDA_TRY(cudaMemcpy(host_points.data(), input_tensor->data(),
                       num_floats * sizeof(float), cudaMemcpyDeviceToHost));
  
  // Apply global rotation to flip Y and Z: [[1,0,0],[0,-1,0],[0,0,-1]]
  for (size_t i = 0; i < num_points; ++i) {
    host_points[i * 3 + 1] = -host_points[i * 3 + 1];  // flip Y
    host_points[i * 3 + 2] = -host_points[i * 3 + 2];  // flip Z
  }
  
  // Add new points to the window trajectory buffer
  for (size_t i = 0; i < num_floats; ++i) {
    window_trajectory_buffer_.push_back(host_points[i]);
  }
  
  // Enforce sliding window: remove oldest points if we exceed max length
  size_t max_floats = kWindowTrajectoryMaxPoints * 3;
  while (window_trajectory_buffer_.size() > max_floats) {
    window_trajectory_buffer_.pop_front();
    window_trajectory_buffer_.pop_front();
    window_trajectory_buffer_.pop_front();
  }
  
  // Copy window trajectory buffer to GPU
  size_t trajectory_size = window_trajectory_buffer_.size();
  if (trajectory_size >= 6) {  // Need at least 2 points for a line
    std::vector<float> trajectory_vec(window_trajectory_buffer_.begin(), window_trajectory_buffer_.end());
    CUDA_TRY(cudaMemcpy(dev_window_trajectory_, trajectory_vec.data(),
                         trajectory_size * sizeof(float), cudaMemcpyHostToDevice));
  }
}

void VizOp::computeTrajectoryBounds() {
  if (trajectory_num_points_ == 0) {
    bounds_min_ = {0.0f, 0.0f, 0.0f};
    bounds_max_ = {0.0f, 0.0f, 0.0f};
    bounds_center_ = {0.0f, 0.0f, 0.0f};
    bounds_radius_ = 1.0f;
    return;
  }
  
  // Initialize with first point
  bounds_min_[0] = bounds_max_[0] = trajectory_host_[0];
  bounds_min_[1] = bounds_max_[1] = trajectory_host_[1];
  bounds_min_[2] = bounds_max_[2] = trajectory_host_[2];
  
  // Find min/max for each axis
  for (size_t i = 1; i < trajectory_num_points_; ++i) {
    float x = trajectory_host_[i * 3 + 0];
    float y = trajectory_host_[i * 3 + 1];
    float z = trajectory_host_[i * 3 + 2];
    
    bounds_min_[0] = std::min(bounds_min_[0], x);
    bounds_min_[1] = std::min(bounds_min_[1], y);
    bounds_min_[2] = std::min(bounds_min_[2], z);
    
    bounds_max_[0] = std::max(bounds_max_[0], x);
    bounds_max_[1] = std::max(bounds_max_[1], y);
    bounds_max_[2] = std::max(bounds_max_[2], z);
  }
  
  // Calculate center
  bounds_center_[0] = (bounds_min_[0] + bounds_max_[0]) * 0.5f;
  bounds_center_[1] = (bounds_min_[1] + bounds_max_[1]) * 0.5f;
  bounds_center_[2] = (bounds_min_[2] + bounds_max_[2]) * 0.5f;
  
  // Calculate bounding sphere radius (half diagonal of bounding box)
  float dx = bounds_max_[0] - bounds_min_[0];
  float dy = bounds_max_[1] - bounds_min_[1];
  float dz = bounds_max_[2] - bounds_min_[2];
  bounds_radius_ = std::sqrt(dx*dx + dy*dy + dz*dz) * 0.5f;
  
  // Ensure minimum radius
  bounds_radius_ = std::max(bounds_radius_, 0.1f);
  
  if (verbose_.get()) {
    std::cout << "Trajectory bounds: min=(" << bounds_min_[0] << ", " << bounds_min_[1] << ", " << bounds_min_[2] 
            << ") max=(" << bounds_max_[0] << ", " << bounds_max_[1] << ", " << bounds_max_[2] 
            << ") radius=" << bounds_radius_ << std::endl;
  }
}

void VizOp::fitCameraToTrajectory() {
  // Set camera target to bounding box center
  camera_target_ = bounds_center_;
  
  // Calculate distance needed to fit the bounding sphere in view
  // Using a simple formula: distance = radius / sin(fov/2)
  // Assuming ~60 degree FOV, sin(30°) ≈ 0.5
  // Add padding factor for some breathing room
  float desired_distance = bounds_radius_ * kFitPadding * 2.0f * user_zoom_factor_;
  camera_distance_ = std::clamp(desired_distance, kMinDistance, kMaxDistance);
  
  if (verbose_.get()) {
    std::cout << "Camera: target=(" << camera_target_[0] << ", " << camera_target_[1] << ", " << camera_target_[2]
            << ") desired_dist=" << desired_distance << " actual_dist=" << camera_distance_ << std::endl;
  }
}

void VizOp::handleCameraInput() {
  viz::BeginImGuiLayer();
  
  ImGuiIO& io = ImGui::GetIO();
  float mouse_x = io.MousePos.x;
  float mouse_y = io.MousePos.y;
  
  // Mouse wheel zoom (adjusts user zoom factor)
  if (io.MouseWheel != 0.0f) {
    user_zoom_factor_ -= io.MouseWheel * kZoomSensitivity;
    user_zoom_factor_ = std::clamp(user_zoom_factor_, kMinZoomFactor, kMaxZoomFactor);
  }
  
  // Mouse drag handling
  if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    if (mouse_dragging_) {
      float dx = mouse_x - last_mouse_x_;
      float dy = mouse_y - last_mouse_y_;
      
      if (io.KeyCtrl) {
        // Ctrl + drag: zoom (adjusts user zoom factor)
        user_zoom_factor_ += dy * kZoomSensitivity * 0.01f;
        user_zoom_factor_ = std::clamp(user_zoom_factor_, kMinZoomFactor, kMaxZoomFactor);
      } else {
        // Regular drag: rotate
        camera_azimuth_ -= dx * kRotateSensitivity;
        camera_elevation_ += dy * kRotateSensitivity;
        camera_elevation_ = std::clamp(camera_elevation_, kMinElevation, kMaxElevation);
      }
    }
    mouse_dragging_ = true;
  } else {
    mouse_dragging_ = false;
  }
  
  // Double-click to reset zoom
  if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    user_zoom_factor_ = 1.0f;
  }
  
  last_mouse_x_ = mouse_x;
  last_mouse_y_ = mouse_y;
  
  viz::EndLayer();
}

void VizOp::updateCamera() {
  // Calculate camera position from spherical coordinates
  float cos_az = std::cos(camera_azimuth_);
  float sin_az = std::sin(camera_azimuth_);
  float cos_el = std::cos(camera_elevation_);
  float sin_el = std::sin(camera_elevation_);
  
  float eye_x = camera_target_[0] + camera_distance_ * cos_el * cos_az;
  float eye_y = camera_target_[1] + camera_distance_ * cos_el * sin_az;
  float eye_z = camera_target_[2] + camera_distance_ * sin_el;
  
  // Up vector (always pointing "up" relative to elevation)
  float up_x = -sin_el * cos_az;
  float up_y = -sin_el * sin_az;
  float up_z = cos_el;
  
  viz::SetCamera(eye_x, eye_y, eye_z,
                 camera_target_[0], camera_target_[1], camera_target_[2],
                 up_x, up_y, up_z);
}

void VizOp::renderTrajectory() {
  
  size_t num_trajectory_points = trajectory_num_points_;
  
  // Draw full trajectory as LINE_STRIP (green-cyan)
  if (num_trajectory_points >= 2) {
    viz::Color(0.0f, 1.0f, 0.5f, 1.0f);  // Green-cyan color
    viz::LineWidth(2.0f);
    
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(dev_trajectory_);
    viz::PrimitiveCudaDevice(viz::PrimitiveTopology::LINE_STRIP_3D, 
                              static_cast<uint32_t>(num_trajectory_points - 1), 
                              3 * num_trajectory_points,
                              dev_ptr);
  }
  
  // Draw the current point as a larger point for visibility
  if (num_trajectory_points >= 1) {
    viz::Color(1.0f, 0.3f, 0.3f, 1.0f);  // Red-ish color for current point
    viz::PointSize(8.0f);
    
    CUdeviceptr dev_ptr_last = reinterpret_cast<CUdeviceptr>(dev_trajectory_) + 
                                (num_trajectory_points - 1) * 3 * sizeof(float);
    viz::PrimitiveCudaDevice(viz::PrimitiveTopology::POINT_LIST, 1, 3, dev_ptr_last);
  }
}

void VizOp::renderWindowTrajectory() {
  
  size_t num_window_points = window_trajectory_buffer_.size() / 3;
  
  // Draw window trajectory as LINE_STRIP (orange/yellow for contrast)
  if (num_window_points >= 2) {
    viz::Color(1.0f, 0.6f, 0.0f, 1.0f);  // Orange color
    viz::LineWidth(3.0f);  // Slightly thicker
    
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(dev_window_trajectory_);
    viz::PrimitiveCudaDevice(viz::PrimitiveTopology::LINE_STRIP_3D, 
                              static_cast<uint32_t>(num_window_points - 1), 
                              3 * num_window_points,
                              dev_ptr);
  }
  
  // Draw the latest point in the window as a bright point
  if (num_window_points >= 1) {
    viz::Color(1.0f, 1.0f, 0.0f, 1.0f);  // Bright yellow for current window point
    viz::PointSize(10.0f);
    
    CUdeviceptr dev_ptr_last = reinterpret_cast<CUdeviceptr>(dev_window_trajectory_) + 
                                (num_window_points - 1) * 3 * sizeof(float);
    viz::PrimitiveCudaDevice(viz::PrimitiveTopology::POINT_LIST, 1, 3, dev_ptr_last);
  }
}

// Functor to flip Y and Z components: (x, y, z) -> (x, -y, -z)
// Applied to floats where index % 3 == 1 (Y) or index % 3 == 2 (Z)
struct FlipYZFunctor {
  __host__ __device__ float operator()(const thrust::tuple<float, size_t>& t) const {
    float val = thrust::get<0>(t);
    size_t idx = thrust::get<1>(t);
    size_t component = idx % 3;
    // Flip Y (component 1) and Z (component 2), keep X (component 0)
    return (component == 0) ? val : -val;
  }
};

void VizOp::renderPointCloud(const std::shared_ptr<holoscan::Tensor>& input_tensor) {
  // Point cloud comes with shape (N, 3)
  auto num_points = input_tensor->shape()[0];
  if (num_points == 0) return;
  
  size_t num_floats = num_points * 3;
  
  // Reallocate GPU buffer if needed
  if (num_floats > dev_pointcloud_capacity_) {
    if (dev_pointcloud_) {
      cudaFree(dev_pointcloud_);
    }
    dev_pointcloud_capacity_ = num_floats * 2;
    CUDA_TRY(cudaMalloc(&dev_pointcloud_, dev_pointcloud_capacity_ * sizeof(float)));
  }
  
  // Get device pointers
  const float* src = static_cast<const float*>(input_tensor->data());
  thrust::device_ptr<const float> src_ptr(src);
  thrust::device_ptr<float> dst_ptr(dev_pointcloud_);
  
  // Transform on GPU: flip Y and Z components
  // Uses counting iterator to track index for component selection
  thrust::transform(
    thrust::make_zip_iterator(thrust::make_tuple(src_ptr, thrust::counting_iterator<size_t>(0))),
    thrust::make_zip_iterator(thrust::make_tuple(src_ptr + num_floats, thrust::counting_iterator<size_t>(num_floats))),
    dst_ptr,
    FlipYZFunctor()
  );
  
  viz::Color(1.0f, 0.8f, 0.2f, 0.5f);  // Semi-transparent (alpha = 0.5)
  viz::PointSize(4.0f);
  viz::PrimitiveCudaDevice(viz::PrimitiveTopology::POINT_LIST_3D, num_points, num_floats,
                            reinterpret_cast<CUdeviceptr>(dev_pointcloud_));
}

void VizOp::renderGroundPlaneGrid() {
  // Render a grid on the XY plane at Z=0 for spatial reference
  // Grid is FIXED at the world origin and grows to encompass the trajectory
  
  if (trajectory_num_points_ == 0) return;
  
  // Grid is always centered at origin (0, 0, 0)
  constexpr float grid_z = 0.0f;
  
  // Calculate grid size to encompass trajectory from origin
  // Find the maximum distance from origin in XY plane
  float max_abs_x = std::max(std::abs(bounds_min_[0]), std::abs(bounds_max_[0]));
  float max_abs_y = std::max(std::abs(bounds_min_[1]), std::abs(bounds_max_[1]));
  float half_size = std::max(max_abs_x, max_abs_y);
  
  // Ensure minimum size and add padding
  half_size = std::max(half_size, 1.0f);
  half_size *= 1.2f;  // 20% padding
  
  // Calculate grid spacing - aim for ~1 unit spacing, with minimum 10 lines
  float spacing = 1.0f;
  int num_lines_half = static_cast<int>(std::ceil(half_size / spacing));
  num_lines_half = std::max(num_lines_half, 5);
  
  // Recalculate half_size to be exact multiple of spacing
  half_size = num_lines_half * spacing;
  int num_lines = num_lines_half * 2 + 1;  // Include center line
  
  // Build grid lines
  std::vector<float> grid_lines;
  grid_lines.reserve(num_lines * 2 * 6);  // Each line: 2 points × 3 coords
  
  // Lines parallel to X axis (varying Y)
  for (int i = -num_lines_half; i <= num_lines_half; ++i) {
    float y = i * spacing;
    grid_lines.push_back(-half_size);  // Start X
    grid_lines.push_back(y);           // Start Y
    grid_lines.push_back(grid_z);      // Start Z
    grid_lines.push_back(half_size);   // End X
    grid_lines.push_back(y);           // End Y
    grid_lines.push_back(grid_z);      // End Z
  }
  
  // Lines parallel to Y axis (varying X)
  for (int i = -num_lines_half; i <= num_lines_half; ++i) {
    float x = i * spacing;
    grid_lines.push_back(x);           // Start X
    grid_lines.push_back(-half_size);  // Start Y
    grid_lines.push_back(grid_z);      // Start Z
    grid_lines.push_back(x);           // End X
    grid_lines.push_back(half_size);   // End Y
    grid_lines.push_back(grid_z);      // End Z
  }
  
  // Render grid with subtle color
  viz::Color(0.4f, 0.4f, 0.5f, 0.3f);  // Gray-blue, semi-transparent
  viz::LineWidth(1.0f);
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 
                  num_lines * 2,  // Number of line segments
                  grid_lines.size(), 
                  grid_lines.data());
  
  // Highlight the X and Y axes through origin (world axes)
  std::vector<float> axis_lines = {
    // X axis (red-ish)
    -half_size, 0.0f, grid_z,
    half_size, 0.0f, grid_z,
    // Y axis (green-ish)
    0.0f, -half_size, grid_z,
    0.0f, half_size, grid_z,
  };
  
  // X axis highlight
  viz::Color(0.7f, 0.3f, 0.3f, 0.6f);  // Reddish
  viz::LineWidth(2.0f);
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 1, 6, axis_lines.data());
  
  // Y axis highlight
  viz::Color(0.3f, 0.7f, 0.3f, 0.6f);  // Greenish
  viz::LineWidth(2.0f);
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 1, 6, axis_lines.data() + 6);
}

void VizOp::renderAxisGizmo() {
  // Gizmo rendered as 2D lines with manually projected coordinates
  // This avoids using SetCamera() which interferes with the main 3D view
  
  // Gizmo position and size in normalized screen coordinates
  constexpr float kGizmoSize = 0.10f;      // Size of gizmo area
  constexpr float kGizmoMargin = 0.02f;    // Margin from edge
  constexpr float kAxisLength = 0.035f;    // Length of axes in screen coords
  
  // Center of gizmo in screen coordinates (bottom-right corner)
  float cx = 1.0f - kGizmoSize / 2.0f - kGizmoMargin;
  float cy = 1.0f - kGizmoSize / 2.0f - kGizmoMargin;
  
  // Build rotation matrix from camera orientation
  // This rotates world axes to view space
  float cos_az = std::cos(camera_azimuth_);
  float sin_az = std::sin(camera_azimuth_);
  float cos_el = std::cos(camera_elevation_);
  float sin_el = std::sin(camera_elevation_);
  
  // View matrix components (simplified orbit camera)
  // Projects 3D world axis to 2D screen
  auto project = [&](float wx, float wy, float wz) -> std::pair<float, float> {
    // Rotate around Z (azimuth), then around X (elevation)
    float x1 = wx * cos_az + wy * sin_az;
    float y1 = -wx * sin_az + wy * cos_az;
    float z1 = wz;
    
    float x2 = x1;
    float y2 = y1 * cos_el - z1 * sin_el;
    
    // Map to screen: X -> right, Y -> up (flip for screen coords)
    return {cx + x2 * kAxisLength, cy - y2 * kAxisLength};
  };
  
  // Project axis endpoints
  auto [ox, oy] = project(0, 0, 0);  // Origin
  auto [xx, xy] = project(1, 0, 0);  // X axis tip
  auto [yx, yy] = project(0, 1, 0);  // Y axis tip
  auto [zx, zy] = project(0, 0, 1);  // Z axis tip
  
  viz::BeginGeometryLayer();
  
  viz::LineWidth(3.0f);
  
  // X axis (Red)
  viz::Color(1.0f, 0.2f, 0.2f, 1.0f);
  std::vector<float> x_axis = {ox, oy, xx, xy};
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST, 1, x_axis.size(), x_axis.data());
  
  // Y axis (Green)
  viz::Color(0.2f, 1.0f, 0.2f, 1.0f);
  std::vector<float> y_axis = {ox, oy, yx, yy};
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST, 1, y_axis.size(), y_axis.data());
  
  // Z axis (Blue)
  viz::Color(0.3f, 0.5f, 1.0f, 1.0f);
  std::vector<float> z_axis = {ox, oy, zx, zy};
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST, 1, z_axis.size(), z_axis.data());
  
  // Axis tip points
  viz::PointSize(8.0f);
  
  viz::Color(1.0f, 0.2f, 0.2f, 1.0f);
  std::vector<float> x_tip = {xx, xy};
  viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, x_tip.size(), x_tip.data());
  
  viz::Color(0.2f, 1.0f, 0.2f, 1.0f);
  std::vector<float> y_tip = {yx, yy};
  viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, y_tip.size(), y_tip.data());
  
  viz::Color(0.3f, 0.5f, 1.0f, 1.0f);
  std::vector<float> z_tip = {zx, zy};
  viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, z_tip.size(), z_tip.data());
  
  // Origin point (white)
  viz::Color(1.0f, 1.0f, 1.0f, 1.0f);
  viz::PointSize(6.0f);
  std::vector<float> origin = {ox, oy};
  viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 1, origin.size(), origin.data());
  
  viz::EndLayer();
}

void VizOp::renderCameraFrustum(const std::shared_ptr<holoscan::Tensor>& camera_position,
                                 const std::shared_ptr<holoscan::Tensor>& camera_rotation) {
  // camera_position is the latest window position (single point)
  // camera_rotation is the latest window rotation (single 3x3 matrix)
  auto num_frames = camera_position->shape()[0];
  if (num_frames == 0) return;
  
  // Copy the latest position and rotation from GPU to host
  // Since camera_position_window is already sliced to [t:t+1], we use index 0
  std::vector<float> host_pos(3);
  std::vector<float> host_rot(9);  // 3x3 rotation matrix, row-major
  
  const float* pos_data = static_cast<const float*>(camera_position->data());
  const float* rot_data = static_cast<const float*>(camera_rotation->data());
  
  // Use first (and only) frame since window tensors are already sliced
  CUDA_TRY(cudaMemcpy(host_pos.data(), 
                       pos_data, 
                       3 * sizeof(float), 
                       cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(host_rot.data(), 
                       rot_data, 
                       9 * sizeof(float), 
                       cudaMemcpyDeviceToHost));
  
  // Apply global rotation to flip Y and Z: [[1,0,0],[0,-1,0],[0,0,-1]]
  // Transform camera center: (x, y, z) -> (x, -y, -z)
  float cx = host_pos[0];
  float cy = -host_pos[1];  // flip Y
  float cz = -host_pos[2];  // flip Z
  
  // Apply global rotation to camera orientation
  // global_rotation @ camera_orientation
  // R_global = [[1,0,0],[0,-1,0],[0,0,-1]]
  // Result: row 0 stays same, row 1 negated, row 2 negated
  // Input is row-major: [r00, r01, r02, r10, r11, r12, r20, r21, r22]
  float r[9];
  r[0] = host_rot[0];  r[1] = host_rot[1];  r[2] = host_rot[2];   // row 0
  r[3] = -host_rot[3]; r[4] = -host_rot[4]; r[5] = -host_rot[5];  // row 1 negated
  r[6] = -host_rot[6]; r[7] = -host_rot[7]; r[8] = -host_rot[8];  // row 2 negated
  
  // Frustum dimensions in world units
  constexpr float kFocalWorldUnits = 1.5f;
  constexpr float kHeightWorldUnits = 2.5f;
  // Assuming 16:9 aspect ratio: width = height * (16/9)
  constexpr float kWidthWorldUnits = 2.5f * (16.0f / 9.0f);
  
  float hw = kWidthWorldUnits / 2.0f;
  float hh = kHeightWorldUnits / 2.0f;
  float f = kFocalWorldUnits;
  
  // Local corners of the image plane (before transformation)
  // Order: bottom-left, bottom-right, top-right, top-left
  float local_corners[4][3] = {
      {-hw, -hh, f},  // square1
      { hw, -hh, f},  // square2
      { hw,  hh, f},  // square3
      {-hw,  hh, f}   // square4
  };
  
  // Transform corners: camera_orientation @ corner + camera_center
  float world_corners[4][3];
  for (int i = 0; i < 4; ++i) {
      float lx = local_corners[i][0];
      float ly = local_corners[i][1];
      float lz = local_corners[i][2];
      
      // Matrix-vector multiply: r @ [lx, ly, lz]
      // r is row-major, so r[row*3 + col]
      world_corners[i][0] = r[0]*lx + r[1]*ly + r[2]*lz + cx;
      world_corners[i][1] = r[3]*lx + r[4]*ly + r[5]*lz + cy;
      world_corners[i][2] = r[6]*lx + r[7]*ly + r[8]*lz + cz;
  }
  
  // Draw lines from apex (camera center) to each corner
  viz::Color(0.33f, 0.33f, 0.33f, 0.75f);  // Gray, semi-transparent (matching Python 0x535353)
  viz::LineWidth(1.0f);
  
  std::vector<float> apex_lines;
  for (int i = 0; i < 4; ++i) {
      // Line from camera center to corner
      apex_lines.push_back(cx);
      apex_lines.push_back(cy);
      apex_lines.push_back(cz);
      apex_lines.push_back(world_corners[i][0]);
      apex_lines.push_back(world_corners[i][1]);
      apex_lines.push_back(world_corners[i][2]);
  }
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 4, apex_lines.size(), apex_lines.data());
  
  // Draw rectangle connecting corners (square base of the pyramid)
  std::vector<float> rect_lines;
  for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      rect_lines.push_back(world_corners[i][0]);
      rect_lines.push_back(world_corners[i][1]);
      rect_lines.push_back(world_corners[i][2]);
      rect_lines.push_back(world_corners[j][0]);
      rect_lines.push_back(world_corners[j][1]);
      rect_lines.push_back(world_corners[j][2]);
  }
  viz::Primitive(viz::PrimitiveTopology::LINE_LIST_3D, 4, rect_lines.size(), rect_lines.data());
}

std::shared_ptr<holoscan::Tensor> VizOp::readTensorMap(const TensorMap& tensormap, const std::string& tensor_name) {
  auto maybe_input_tensor = tensormap.find(tensor_name);
  if (maybe_input_tensor == tensormap.end()) {
    throw std::runtime_error(
        fmt::format("Operator '{}' failed to find '{}' tensor", name_, tensor_name));
  }
  
  auto input_tensor = maybe_input_tensor->second;
  if (!input_tensor->is_contiguous()) {
    throw std::runtime_error("Input tensor must have row-major memory layout.");
  }
  
  return input_tensor;
}

void VizOp::renderFramePoints(const std::shared_ptr<holoscan::Tensor>& frame, const std::shared_ptr<holoscan::Tensor>& point_coords) {
  // Render frame. It comes with shape (H, W, 3)
  viz::BeginImageLayer();
  viz::LayerAddView(0.0, 0.0, left_side_ratio_, 1.0);
  // ImageCudaDevice takes width, height, format, and device pointer
  viz::ImageCudaDevice(256, 256, viz::ImageFormat::R8G8B8_UNORM, reinterpret_cast<CUdeviceptr>(frame->data()));
  viz::EndLayer();

  // Render point cloud. It comes with shape (N, 2)
  auto num_points = point_coords->shape()[0];
  viz::BeginGeometryLayer();
  viz::LayerAddView(0.0, 0.0, left_side_ratio_, 1.0);
  viz::Color(1.0f, 0.8f, 0.2f, 1.0f);
  viz::PointSize(5.5f);
  viz::PrimitiveCudaDevice(viz::PrimitiveTopology::POINT_LIST, num_points, num_points * 2, reinterpret_cast<CUdeviceptr>(point_coords->data()));
  viz::EndLayer();
}

void VizOp::compute(InputContext& op_input, OutputContext& op_output,
                    ExecutionContext& context) {
  // Receive and validate input
  auto maybe_tensormap = op_input.receive<TensorMap>("receivers");
  if (!maybe_tensormap) {
    throw std::runtime_error(
        fmt::format("Operator '{}' failed to receive input on 'receivers': {}",
                    name_, maybe_tensormap.error().what()));
  }
  
  auto& tensormap = maybe_tensormap.value();
  auto input_camera_position = readTensorMap(tensormap, "camera_position");
  auto input_camera_position_window = readTensorMap(tensormap, "camera_position_window");
  auto input_camera_rotation = readTensorMap(tensormap, "camera_rotation");
  auto input_3d_points = readTensorMap(tensormap, "points3D");
  auto input_frame = readTensorMap(tensormap, "frame");
  auto input_2d_points = readTensorMap(tensormap, "point_coords");

  // Process incoming trajectory points (full trajectory each frame)
  processIncomingTrajectory(input_camera_position);
  
  // Process window trajectory (sliding window of last N points)
  processWindowTrajectory(input_camera_position_window);
  
  // Compute bounding box and fit camera
  computeTrajectoryBounds();
  fitCameraToTrajectory();

  // Render frame
  viz::Begin();
  
  handleCameraInput();
  
  // Main 3D view layer - camera must be set INSIDE the layer
  viz::BeginGeometryLayer();
  viz::LayerAddView(left_side_ratio_, 0.0, 1.0 - left_side_ratio_, 1.0);
  updateCamera();  // Set camera inside the layer context
  renderGroundPlaneGrid();      // Ground reference grid (render first, behind other elements)
  renderTrajectory();           // Full trajectory (green-cyan)
  renderWindowTrajectory();     // Window trajectory (orange)
  renderCameraFrustum(input_camera_position_window, input_camera_rotation);  // Frustum at latest window position
  renderPointCloud(input_3d_points);
  viz::EndLayer();

  renderFramePoints(input_frame, input_2d_points);
  
  // Render axis gizmo AFTER main layer (overlays on top)
  // Uses 2D rendering with manual projection - doesn't affect 3D camera
  renderAxisGizmo();

  viz::End();
}

} // namespace holoscan::ops
