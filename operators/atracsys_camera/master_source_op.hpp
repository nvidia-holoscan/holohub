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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

extern "C" {
typedef struct CUstream_st* cudaStream_t;
}

#include "holoscan/holoscan.hpp"

#include "hardware_mode_command.hpp"
#include "sdk/atracsys_device.hpp"
#include "sdk/s3dk_interface.hpp"
#include "sdk/sdk_wrapper.hpp"

namespace holoscan::ops {

class __attribute__((visibility("default"))) AtracsysMasterSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(AtracsysMasterSourceOp, holoscan::Operator)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  enum class SchedulerMode {
    kMixed,
    kExclusive,
  };

  void reset_state();
  void capture_loop();
  void load_geometries();
  void configure_camera();
  void configure_frame();
  void destroy_frame();
  void ensure_visible_output_entities(const holoscan::ExecutionContext& context, uint32_t width,
                                      uint32_t height);
  void ensure_ir_output_entities(const holoscan::ExecutionContext& context, uint32_t width,
                                 uint32_t height);
  void ensure_marker_poses_output_entities(const holoscan::ExecutionContext& context);
  void ensure_disparity_output_entities(const holoscan::ExecutionContext& context, uint32_t width,
                                        uint32_t height);
  void configure_structured_frame_state() const;
  void emit_visible_frame(holoscan::OutputContext& op_output, holoscan::ExecutionContext& context,
                          cudaStream_t cuda_stream);
  void emit_ir_frame(holoscan::OutputContext& op_output, holoscan::ExecutionContext& context,
                     cudaStream_t cuda_stream);
  void emit_marker_poses(holoscan::OutputContext& op_output, holoscan::ExecutionContext& context,
                         cudaStream_t cuda_stream);
  void emit_structured_points(holoscan::OutputContext& op_output,
                              holoscan::ExecutionContext& context, cudaStream_t cuda_stream);
  void apply_pending_command(const std::shared_ptr<atracsys::HardwareModeCommand>& cmd);
  void set_scheduler_pattern(const std::string& pattern);
  void set_exclusive_pattern(atracsys::HardwareMode mode);
  std::string configured_mixed_pattern() const;
  bool upload_frame_to_base_tensor(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                                   const char* frame_kind, cudaStream_t cuda_stream);
  SchedulerMode configured_scheduler_mode() const;
  atracsys::HardwareMode configured_initial_hw_mode() const;

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> image_allocator_;
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> structured_allocator_;
  holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;
  holoscan::Parameter<std::string> geometry_path_;
  holoscan::Parameter<int> vis_integration_time_us_;
  holoscan::Parameter<int> sl_integration_time_us_;
  holoscan::Parameter<int> scale_factor_;
  holoscan::Parameter<std::string> scheduler_mode_;
  holoscan::Parameter<std::string> initial_hw_mode_;
  holoscan::Parameter<bool> enable_visible_;
  holoscan::Parameter<bool> enable_ir_;
  holoscan::Parameter<bool> enable_structured_;

  static constexpr size_t kEntityRingSize = 4;
  static constexpr size_t kMaxMarkers = 10;

  RealSDKWrapper sdk_;
  std::unique_ptr<RealS3DKWrapper> s3dk_;
  uint64_t device_sn_{0};
  ftkFrameQuery* frame_{nullptr};

  StereoParameters* stereo_params_{nullptr};
  stereo_matching_engine* engine_{nullptr};
  GpuFrame3D* gpu_frame_{nullptr};
  ImageType3D image_type_{};

  ftkRigidBody geometry_{};

  std::vector<float> marker_poses_;

  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> visible_output_entities_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> ir_output_entities_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> marker_poses_output_entities_;
  std::array<std::optional<holoscan::gxf::Entity>, kEntityRingSize> disparity_output_entities_;

  size_t visible_output_entity_index_{0};
  size_t ir_output_entity_index_{0};
  size_t marker_poses_output_entity_index_{0};
  size_t disparity_output_entity_index_{0};

  uint32_t visible_output_width_{0};
  uint32_t visible_output_height_{0};
  uint32_t ir_output_width_{0};
  uint32_t ir_output_height_{0};
  uint32_t disparity_output_width_{0};
  uint32_t disparity_output_height_{0};

  SchedulerMode active_scheduler_mode_{SchedulerMode::kMixed};
  atracsys::HardwareMode active_hw_mode_{atracsys::HardwareMode::kVisible};
  std::string current_pattern_;
  uint32_t frame_timeout_ms_{50};
  uint64_t frame_timeout_count_{0};
  bool first_frame_logged_{false};
  bool first_structured_cloud_logged_{false};

  std::shared_ptr<holoscan::AsynchronousCondition> async_cond_;
  std::atomic<bool> is_running_{false};
  std::thread capture_thread_;
};

}  // namespace holoscan::ops
