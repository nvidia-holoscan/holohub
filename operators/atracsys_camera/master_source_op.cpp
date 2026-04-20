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

#include "master_source_op.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include <s3dk_gpu.hpp>

#include "gxf/std/allocator.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/logger/logger.hpp"
#include "point_cloud_filter.cuh"
#include "sdk/s3dk_wrapper.hpp"

namespace holoscan::ops {

namespace {

constexpr const char* kBaseTensorName = "base";
constexpr const char* kMarkerPosesTensorName = "marker_poses";
constexpr size_t kMaxRawFiducials = 32;
constexpr size_t kMaxTriangulatedFiducials = 16;

inline void check_cuda(cudaError_t code, const char* message) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(code));
  }
}

char slot_for_mode(atracsys::HardwareMode mode) {
  switch (mode) {
    case atracsys::HardwareMode::kVisible:
      return 'V';
    case atracsys::HardwareMode::kInfrared:
      return 'I';
    case atracsys::HardwareMode::kStructured:
      return 'S';
    default:
      return 'V';
  }
}

const char* mode_name(atracsys::HardwareMode mode) {
  switch (mode) {
    case atracsys::HardwareMode::kVisible:
      return "visible";
    case atracsys::HardwareMode::kInfrared:
      return "infrared";
    case atracsys::HardwareMode::kStructured:
      return "structured";
    default:
      return "visible";
  }
}

bool is_visible_format(ftkPixelFormat format) {
  return format == ftkPixelFormat::GRAY8_VIS || format == ftkPixelFormat::GRAY16_VIS;
}

bool is_ir_format(ftkPixelFormat format) {
  return format == ftkPixelFormat::GRAY8 || format == ftkPixelFormat::GRAY16;
}

bool is_structured_format(ftkPixelFormat format) {
  return format == ftkPixelFormat::GRAY8_SL || format == ftkPixelFormat::GRAY16_SL;
}

const char* frame_format_name(ftkPixelFormat format) {
  if (is_visible_format(format)) {
    return "visible";
  }
  if (is_ir_format(format)) {
    return "infrared";
  }
  if (is_structured_format(format)) {
    return "structured";
  }
  return "unknown";
}

int validated_scale_factor(int requested_scale) {
  switch (requested_scale) {
    case 1:
    case 2:
    case 4:
    case 8:
      return requested_scale;
    default:
      HOLOSCAN_LOG_WARN(
          "AtracsysMasterSourceOp: unsupported structured-light scale {} requested, "
          "defaulting to 1",
          requested_scale);
      return 1;
  }
}

}  // namespace

void AtracsysMasterSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<std::shared_ptr<atracsys::HardwareModeCommand>>("in_hw_cmd")
      .condition(holoscan::ConditionType::kNone);

  spec.output<holoscan::gxf::Entity>("out_visible_base");
  spec.output<holoscan::gxf::Entity>("out_ir_base");
  spec.output<holoscan::gxf::Entity>("out_marker_poses");
  spec.output<holoscan::gxf::Entity>("out_disparity");
  spec.output<std::shared_ptr<std::vector<float>>>("out_q_matrix");

  spec.param(image_allocator_,
             "image_allocator",
             "ImageAllocator",
             "Visible and infrared frame output allocator");
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CudaStreamPool",
             "CUDA stream pool used for async frame uploads");
  spec.param(structured_allocator_,
             "structured_allocator",
             "StructuredAllocator",
             "Structured light output allocator");
  spec.param(geometry_path_,
             "geometry_path",
             "GeometryPath",
             "Path to the rigid-body geometry .ini file",
             std::string("geometry10.ini"));
  spec.param(vis_integration_time_us_,
             "vis_integration_time_us",
             "VisIntegrationTimeUs",
             "Integration time in microseconds for visible frames",
             16000);
  spec.param(sl_integration_time_us_,
             "sl_integration_time_us",
             "StructuredIntegrationTimeUs",
             "Integration time in microseconds for structured light frames",
             16000);
  spec.param(scale_factor_, "scale", "Scale", "Scale factor (1, 2, 4, 8)", 1);
  spec.param(scheduler_mode_,
             "scheduler_mode",
             "SchedulerMode",
             "mixed (concurrent VSI) or exclusive (single mode)",
             std::string("mixed"));
  spec.param(initial_hw_mode_,
             "initial_hw_mode",
             "InitialHardwareMode",
             "Initial hardware mode visible|ir|structured",
             std::string("visible"));
  spec.param(enable_visible_,
             "enable_visible",
             "EnableVisible",
             "Whether visible frames are enabled in the live-camera schedule",
             true);
  spec.param(enable_ir_,
             "enable_ir",
             "EnableIr",
             "Whether infrared frames are enabled in the live-camera schedule",
             true);
  spec.param(enable_structured_,
             "enable_structured",
             "EnableStructured",
             "Whether structured-light frames are enabled in the live-camera schedule",
             true);
}

void AtracsysMasterSourceOp::start() {
  reset_state();

  for (auto& [name, cond] : conditions()) {
    if (auto async_cond = std::dynamic_pointer_cast<holoscan::AsynchronousCondition>(cond)) {
      async_cond_ = async_cond;
      break;
    }
  }
  if (!async_cond_) {
    throw std::runtime_error(
        "AtracsysMasterSourceOp: AsyncCondition not found. The app must provide an "
        "AsynchronousCondition.");
  }

  frame_timeout_count_ = 0;
  first_frame_logged_ = false;
  first_structured_cloud_logged_ = false;

  auto& dev = AtracsysDevice::instance();
  dev.init();

  try {
    device_sn_ = dev.serial();

    active_scheduler_mode_ = configured_scheduler_mode();
    active_hw_mode_ = configured_initial_hw_mode();
    frame_timeout_ms_ = 500;

    if (!s3dk_) {
      s3dk_ = std::make_unique<RealS3DKWrapper>();
    }
    if (!s3dk_->initializeDeviceHelper(&device_sn_, dev.lib(), &image_type_)) {
      throw std::runtime_error(
          "AtracsysMasterSourceOp: initializeDeviceHelper failed");
    }

    load_geometries();
    configure_camera();

    frame_ = sdk_.createFrame();
    if (!frame_) {
      throw std::runtime_error(
          "AtracsysMasterSourceOp: ftkCreateFrame failed");
    }
    configure_frame();

    stereo_params_ = s3dk_->createStereoParameters();
    engine_ = s3dk_->createDefaultEngine();
    gpu_frame_ = s3dk_->createGpu3DFrame(image_type_);
    if (gpu_frame_) {
      const int scale = validated_scale_factor(scale_factor_.get());
      gpu_frame_->_scale = scale;
    }
    if (stereo_params_) {
      stereo_params_->scale = validated_scale_factor(scale_factor_.get());
    }

    is_running_ = true;
    async_cond_->event_state(
        holoscan::AsynchronousEventState::EVENT_WAITING);
    capture_thread_ =
        std::thread(&AtracsysMasterSourceOp::capture_loop, this);
  } catch (...) {
    AtracsysDevice::instance().shutdown();
    throw;
  }
}

void AtracsysMasterSourceOp::stop() {
  is_running_ = false;
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }

  reset_state();
  destroy_frame();
  gpu_frame_ = nullptr;
  engine_ = nullptr;
  stereo_params_ = nullptr;
  s3dk_.reset();

  AtracsysDevice::instance().shutdown();

  holoscan::Operator::stop();
}

void AtracsysMasterSourceOp::capture_loop() {
  const auto& dev = AtracsysDevice::instance();

  while (is_running_) {
    if (async_cond_->event_state() == holoscan::AsynchronousEventState::EVENT_DONE) {
      std::this_thread::yield();
      continue;
    }

    const auto frame_status = sdk_.getLastFrame(dev.lib(), dev.serial(), frame_, frame_timeout_ms_);
    if (frame_status == ftkError::FTK_OK && frame_->imageHeader) {
      // The capture thread writes to frame_ here. Below we signal EVENT_DONE, which uses
      // the necessary acquire/release memory ordering (via AsynchronousCondition::event_state)
      // so compute() safely reads frame_ after seeing EVENT_DONE.
      async_cond_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    } else {
      ++frame_timeout_count_;
      if (frame_timeout_count_ == 1 || (frame_timeout_count_ % 100) == 0) {
        HOLOSCAN_LOG_WARN(
            "AtracsysMasterSourceOp: waiting for frames from camera (timeouts={}, timeout_ms={}, "
            "status={})",
            frame_timeout_count_,
            frame_timeout_ms_,
            static_cast<int>(frame_status));
      }
    }
  }
}

void AtracsysMasterSourceOp::compute(holoscan::InputContext& op_input,
                                     holoscan::OutputContext& op_output,
                                     holoscan::ExecutionContext& context) {
  auto hw_cmd = op_input.receive<std::shared_ptr<atracsys::HardwareModeCommand>>("in_hw_cmd");
  if (hw_cmd && hw_cmd.value()) {
    if (active_scheduler_mode_ == SchedulerMode::kExclusive) {
      apply_pending_command(hw_cmd.value());
    } else if (active_scheduler_mode_ == SchedulerMode::kMixed) {
      HOLOSCAN_LOG_DEBUG("AtracsysMasterSourceOp: hw_cmd ignored in mixed mode");
    }
  }

  if (!frame_ || !frame_->imageHeader) {
    if (async_cond_)
      async_cond_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    return;
  }

  cudaStream_t cuda_stream = cudaStreamDefault;
  const auto maybe_stream = context.allocate_cuda_stream("atracsys_master_stream");
  if (maybe_stream) {
    cuda_stream = maybe_stream.value();
  }

  const auto format = frame_->imageHeader->format;
  if (!first_frame_logged_) {
    first_frame_logged_ = true;
    HOLOSCAN_LOG_INFO("AtracsysMasterSourceOp: received first {} frame ({}x{})",
                      frame_format_name(format),
                      frame_->imageHeader->width,
                      frame_->imageHeader->height);
  }
  if (is_visible_format(format)) {
    emit_visible_frame(op_output, context, cuda_stream);
  } else if (is_ir_format(format)) {
    emit_ir_frame(op_output, context, cuda_stream);
    emit_marker_poses(op_output, context, cuda_stream);
  } else if (is_structured_format(format)) {
    emit_structured_points(op_output, context, cuda_stream);
  }

  if (async_cond_) {
    async_cond_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
  }
}

void AtracsysMasterSourceOp::reset_state() {
  for (auto& entity : visible_output_entities_) { entity.reset(); }
  for (auto& entity : ir_output_entities_) { entity.reset(); }
  for (auto& entity : marker_poses_output_entities_) { entity.reset(); }
  for (auto& entity : disparity_output_entities_) { entity.reset(); }

  visible_output_entity_index_ = 0;
  ir_output_entity_index_ = 0;
  marker_poses_output_entity_index_ = 0;
  disparity_output_entity_index_ = 0;
  visible_output_width_ = 0;
  visible_output_height_ = 0;
  ir_output_width_ = 0;
  ir_output_height_ = 0;
  disparity_output_width_ = 0;
  disparity_output_height_ = 0;
  marker_poses_.clear();
  current_pattern_.clear();
}

void AtracsysMasterSourceOp::load_geometries() {
  const auto& dev = AtracsysDevice::instance();
  const int ref_status = sdk_.loadBody(dev.lib(), geometry_path_.get(), geometry_);
  if (ref_status == 0 || ref_status == 1) {
    sdk_.setRigidBody(dev.lib(), dev.serial(), &geometry_);
  } else {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: failed to load geometry from {}",
                      geometry_path_.get());
  }
}

void AtracsysMasterSourceOp::configure_camera() {
  const auto& dev = AtracsysDevice::instance();
  const auto& opts = dev.options();

  if (sdk_.setInt32(dev.lib(), dev.serial(), opts.at("Enable embedded processing"), 1) !=
      ftkError::FTK_OK) {
    throw std::runtime_error("AtracsysMasterSourceOp: failed to enable embedded processing");
  }
  if (sdk_.setInt32(dev.lib(), dev.serial(), opts.at("Enable images sending"), 1) !=
      ftkError::FTK_OK) {
    throw std::runtime_error("AtracsysMasterSourceOp: failed to enable image sending");
  }
  if (sdk_.setInt32(dev.lib(),
                    dev.serial(),
                    opts.at("Image Integration Time for VIS frames"),
                    vis_integration_time_us_.get()) != ftkError::FTK_OK) {
    throw std::runtime_error("AtracsysMasterSourceOp: failed to set VIS integration time");
  }
  if (sdk_.setInt32(dev.lib(),
                    dev.serial(),
                    opts.at("Image Integration Time for SL frames"),
                    sl_integration_time_us_.get()) != ftkError::FTK_OK) {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: failed to set SL integration time");
  }
  if (opts.count("Enable dot projectors") &&
      sdk_.setInt32(dev.lib(), dev.serial(), opts.at("Enable dot projectors"), 2) !=
          ftkError::FTK_OK) {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: failed to enable dot projectors");
  }

  if (active_scheduler_mode_ == SchedulerMode::kMixed) {
    set_scheduler_pattern(configured_mixed_pattern());
  } else {
    set_exclusive_pattern(active_hw_mode_);
  }
}

void AtracsysMasterSourceOp::configure_frame() {
  if (sdk_.setFrameOptions(
          true, 20u, kMaxRawFiducials, kMaxRawFiducials, kMaxTriangulatedFiducials, 10u, frame_) !=
      ftkError::FTK_OK) {
    throw std::runtime_error("AtracsysMasterSourceOp: ftkSetFrameOptions failed");
  }
}

void AtracsysMasterSourceOp::destroy_frame() {
  if (frame_) {
    sdk_.destroyFrame(frame_);
    frame_ = nullptr;
  }
}

void AtracsysMasterSourceOp::ensure_visible_output_entities(
    const holoscan::ExecutionContext& context, uint32_t width, uint32_t height) {
  if (visible_output_width_ != width || visible_output_height_ != height) {
    for (auto& entity : visible_output_entities_) { entity.reset(); }
    visible_output_width_ = width;
    visible_output_height_ = height;
    visible_output_entity_index_ = 0;
  }
  if (visible_output_entities_[visible_output_entity_index_]) {
    return;
  }

  const nvidia::gxf::Shape shape{static_cast<int32_t>(height), static_cast<int32_t>(width), 1};
  auto alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), image_allocator_.get()->gxf_cid());
  auto msg = nvidia::gxf::Entity::New(context.context());
  auto tensor = msg.value().add<nvidia::gxf::Tensor>(kBaseTensorName);
  tensor.value()->reshape<uint8_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc.value());
  visible_output_entities_[visible_output_entity_index_].emplace(std::move(msg.value()));
}

void AtracsysMasterSourceOp::ensure_ir_output_entities(const holoscan::ExecutionContext& context,
                                                       uint32_t width, uint32_t height) {
  if (ir_output_width_ != width || ir_output_height_ != height) {
    for (auto& entity : ir_output_entities_) { entity.reset(); }
    ir_output_width_ = width;
    ir_output_height_ = height;
    ir_output_entity_index_ = 0;
  }
  if (ir_output_entities_[ir_output_entity_index_]) {
    return;
  }

  const nvidia::gxf::Shape shape{static_cast<int32_t>(height), static_cast<int32_t>(width), 1};
  auto alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), image_allocator_.get()->gxf_cid());
  auto msg = nvidia::gxf::Entity::New(context.context());
  auto tensor = msg.value().add<nvidia::gxf::Tensor>(kBaseTensorName);
  tensor.value()->reshape<uint8_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc.value());
  ir_output_entities_[ir_output_entity_index_].emplace(std::move(msg.value()));
}

void AtracsysMasterSourceOp::ensure_marker_poses_output_entities(
    const holoscan::ExecutionContext& context) {
  if (marker_poses_output_entities_[marker_poses_output_entity_index_]) {
    return;
  }

  const nvidia::gxf::Shape shape{static_cast<int32_t>(kMaxMarkers), 16};
  auto alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), structured_allocator_.get()->gxf_cid());
  auto msg = nvidia::gxf::Entity::New(context.context());
  auto tensor = msg.value().add<nvidia::gxf::Tensor>(kMarkerPosesTensorName);
  tensor.value()->reshape<float>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc.value());
  marker_poses_output_entities_[marker_poses_output_entity_index_].emplace(std::move(msg.value()));
}

void AtracsysMasterSourceOp::ensure_disparity_output_entities(
    const holoscan::ExecutionContext& context, uint32_t width, uint32_t height) {
  if (disparity_output_width_ != width || disparity_output_height_ != height) {
    for (auto& entity : disparity_output_entities_) { entity.reset(); }
    disparity_output_width_ = width;
    disparity_output_height_ = height;
    disparity_output_entity_index_ = 0;
  }
  if (disparity_output_entities_[disparity_output_entity_index_]) {
    return;
  }

  const nvidia::gxf::Shape shape{static_cast<int32_t>(height), static_cast<int32_t>(width), 1};
  auto alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), structured_allocator_.get()->gxf_cid());
  auto msg = nvidia::gxf::Entity::New(context.context());
  auto tensor = msg.value().add<nvidia::gxf::Tensor>("disparity_map");
  tensor.value()->reshape<int16_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, alloc.value());
  disparity_output_entities_[disparity_output_entity_index_].emplace(std::move(msg.value()));
}

void AtracsysMasterSourceOp::configure_structured_frame_state() const {
  if (!gpu_frame_ || !frame_ || !frame_->imageHeader) {
    return;
  }

  const int width = static_cast<int>(frame_->imageHeader->width);
  const int height = static_cast<int>(frame_->imageHeader->height);
  auto& roi = gpu_frame_->_ROIbox;

  const bool roi_invalid = (roi.x_min < 0) || (roi.y_min < 0) || (roi.x_max <= roi.x_min) ||
                           (roi.y_max <= roi.y_min) || (roi.x_max > width) || (roi.y_max > height);
  if (roi_invalid) {
    roi.x_min = 0;
    roi.y_min = 0;
    roi.x_max = width;
    roi.y_max = height;
  }
}

void AtracsysMasterSourceOp::emit_visible_frame(holoscan::OutputContext& op_output,
                                                holoscan::ExecutionContext& context,
                                                cudaStream_t cuda_stream) {
  const auto& h = *frame_->imageHeader;
  ensure_visible_output_entities(context, h.width, h.height);

  auto& entity = visible_output_entities_[visible_output_entity_index_].value();
  auto gxf_entity = nvidia::gxf::Entity(entity);
  auto tensor = gxf_entity.get<nvidia::gxf::Tensor>(kBaseTensorName);
  if (!upload_frame_to_base_tensor(tensor.value(), "visible", cuda_stream)) {
    return;
  }

  op_output.set_cuda_stream(cuda_stream, "out_visible_base");
  holoscan::gxf::Entity out(entity);
  op_output.emit(out, "out_visible_base");
  visible_output_entities_[visible_output_entity_index_].reset();
  visible_output_entity_index_ =
      (visible_output_entity_index_ + 1) % visible_output_entities_.size();
}

void AtracsysMasterSourceOp::emit_ir_frame(holoscan::OutputContext& op_output,
                                           holoscan::ExecutionContext& context,
                                           cudaStream_t cuda_stream) {
  const auto& h = *frame_->imageHeader;
  ensure_ir_output_entities(context, h.width, h.height);

  auto& entity = ir_output_entities_[ir_output_entity_index_].value();
  auto gxf_entity = nvidia::gxf::Entity(entity);
  auto tensor = gxf_entity.get<nvidia::gxf::Tensor>(kBaseTensorName);
  if (!upload_frame_to_base_tensor(tensor.value(), "infrared", cuda_stream)) {
    return;
  }

  op_output.set_cuda_stream(cuda_stream, "out_ir_base");
  holoscan::gxf::Entity out(entity);
  op_output.emit(out, "out_ir_base");
  ir_output_entities_[ir_output_entity_index_].reset();
  ir_output_entity_index_ = (ir_output_entity_index_ + 1) % ir_output_entities_.size();
}

void AtracsysMasterSourceOp::emit_marker_poses(holoscan::OutputContext& op_output,
                                               holoscan::ExecutionContext& context,
                                               cudaStream_t cuda_stream) {
  ensure_marker_poses_output_entities(context);

  marker_poses_.assign(kMaxMarkers * 16, 0.0F);
  if (frame_->markersCount > 0 && frame_->markers) {
    for (uint32_t i = 0; i < frame_->markersCount && i < kMaxMarkers; ++i) {
      const auto& marker = frame_->markers[i];
      const size_t offset = i * 16;

      marker_poses_[offset + 0] = marker.rotation[0][0];
      marker_poses_[offset + 1] = marker.rotation[1][0];
      marker_poses_[offset + 2] = marker.rotation[2][0];
      marker_poses_[offset + 3] = 0.0f;
      marker_poses_[offset + 4] = marker.rotation[0][1];
      marker_poses_[offset + 5] = marker.rotation[1][1];
      marker_poses_[offset + 6] = marker.rotation[2][1];
      marker_poses_[offset + 7] = 0.0f;
      marker_poses_[offset + 8] = marker.rotation[0][2];
      marker_poses_[offset + 9] = marker.rotation[1][2];
      marker_poses_[offset + 10] = marker.rotation[2][2];
      marker_poses_[offset + 11] = 0.0f;
      marker_poses_[offset + 12] = marker.translationMM[0] / 1000.0f;
      marker_poses_[offset + 13] = marker.translationMM[1] / 1000.0f;
      marker_poses_[offset + 14] = marker.translationMM[2] / 1000.0f;
      marker_poses_[offset + 15] = 1.0f;
    }
  }

  auto& entity = marker_poses_output_entities_[marker_poses_output_entity_index_].value();
  auto gxf_entity = nvidia::gxf::Entity(entity);
  auto marker_poses_tensor = gxf_entity.get<nvidia::gxf::Tensor>(kMarkerPosesTensorName);

  check_cuda(cudaMemcpyAsync(marker_poses_tensor.value()->pointer(),
                             marker_poses_.data(),
                             marker_poses_.size() * sizeof(float),
                             cudaMemcpyHostToDevice,
                             cuda_stream),
             "AtracsysMasterSourceOp: failed to upload marker poses");

  op_output.set_cuda_stream(cuda_stream, "out_marker_poses");
  holoscan::gxf::Entity out(entity);
  op_output.emit(out, "out_marker_poses");
  marker_poses_output_entities_[marker_poses_output_entity_index_].reset();
  marker_poses_output_entity_index_ =
      (marker_poses_output_entity_index_ + 1) % marker_poses_output_entities_.size();
}

void AtracsysMasterSourceOp::emit_structured_points(holoscan::OutputContext& op_output,
                                                    holoscan::ExecutionContext& context,
                                                    cudaStream_t cuda_stream) {
  if (!s3dk_ || !gpu_frame_ || !frame_ || !frame_->imageHeader || !stereo_params_ || !engine_) {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: structured-light path not fully initialized");
    return;
  }
  if (!frame_->imageLeftPixels || !frame_->imageRightPixels) {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: structured frame missing stereo pixels");
    return;
  }

  configure_structured_frame_state();

  try {
    if (!s3dk_->computeDispMap(frame_, engine_, gpu_frame_, stereo_params_)) {
      HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: computeDispMap failed");
      return;
    }
  } catch (const cv::Exception& e) {
    HOLOSCAN_LOG_ERROR("AtracsysMasterSourceOp: cv exception: {}", e.what());
    return;
  }

  auto& disp_map = gpu_frame_->_d_disparityMap;
  if (disp_map.empty() || disp_map.type() != CV_16S) {
    return;
  }

  const int width = disp_map.cols;
  const int height = disp_map.rows;
  auto Q_mat = stereo_params_->Q_32F;
  if (Q_mat.empty() || Q_mat.type() != CV_32FC1 || Q_mat.rows != 4 ||
      Q_mat.cols != 4 || !Q_mat.isContinuous()) {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: invalid Q_matrix structure");
    return;
  }

  ensure_disparity_output_entities(context, width, height);

  auto& entity = disparity_output_entities_[disparity_output_entity_index_].value();
  auto gxf_entity = nvidia::gxf::Entity(entity);
  auto tensor = gxf_entity.get<nvidia::gxf::Tensor>("disparity_map");

  check_cuda(cudaMemcpy2DAsync(tensor.value()->pointer(),
                               width * sizeof(int16_t),
                               disp_map.ptr<int16_t>(),
                               disp_map.step,
                               width * sizeof(int16_t),
                               height,
                               cudaMemcpyDeviceToDevice,
                               cuda_stream),
             "AtracsysMasterSourceOp: failed to upload disparity map");

  op_output.set_cuda_stream(cuda_stream, "out_disparity");
  holoscan::gxf::Entity out(entity);
  op_output.emit(out, "out_disparity");
  disparity_output_entities_[disparity_output_entity_index_].reset();
  disparity_output_entity_index_ =
      (disparity_output_entity_index_ + 1) % disparity_output_entities_.size();

  auto q_msg = std::make_shared<std::vector<float>>(16);
  std::memcpy(q_msg->data(), Q_mat.ptr<float>(), 16 * sizeof(float));
  op_output.emit(q_msg, "out_q_matrix");
}

void AtracsysMasterSourceOp::apply_pending_command(
    const std::shared_ptr<atracsys::HardwareModeCommand>& cmd) {
  if (!cmd || cmd->mode == active_hw_mode_) {
    return;
  }

  active_hw_mode_ = cmd->mode;
  set_exclusive_pattern(active_hw_mode_);
}

void AtracsysMasterSourceOp::set_scheduler_pattern(const std::string& pattern) {
  if (pattern.empty() || pattern == current_pattern_) {
    return;
  }

  auto& dev = AtracsysDevice::instance();
  const auto& opts = dev.options();
  auto& buf = dev.buffer();

  std::memset(&buf, 0, sizeof(buf));
  std::memcpy(buf.data, pattern.c_str(), pattern.size());
  buf.size = static_cast<uint32_t>(pattern.size());

  if (sdk_.setData(dev.lib(), dev.serial(), opts.at("Image Scheduler Pattern"), &buf) !=
      ftkError::FTK_OK) {
    throw std::runtime_error("AtracsysMasterSourceOp: failed to set scheduler pattern");
  }

  current_pattern_ = pattern;
}

void AtracsysMasterSourceOp::set_exclusive_pattern(atracsys::HardwareMode mode) {
  set_scheduler_pattern(std::string(54, slot_for_mode(mode)));
}

std::string AtracsysMasterSourceOp::configured_mixed_pattern() const {
  std::string base_pattern;
  if (enable_visible_.get()) {
    base_pattern.push_back('V');
  }
  if (enable_ir_.get()) {
    base_pattern.push_back('I');
  }
  if (enable_structured_.get()) {
    base_pattern.push_back('S');
  }

  if (base_pattern.empty()) {
    HOLOSCAN_LOG_WARN(
        "AtracsysMasterSourceOp: all live frame types are disabled; defaulting mixed "
        "schedule to visible");
    base_pattern = "V";
  }

  std::string repeated_pattern;
  repeated_pattern.reserve(54);
  while (repeated_pattern.size() < 54) { repeated_pattern += base_pattern; }
  repeated_pattern.resize(54);
  return repeated_pattern;
}

bool AtracsysMasterSourceOp::upload_frame_to_base_tensor(
    const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor, const char* frame_kind,
    cudaStream_t cuda_stream) {
  if (!frame_->imageHeader || !frame_->imageLeftPixels) {
    return false;
  }

  const auto& h = *frame_->imageHeader;
  const size_t row_bytes = static_cast<size_t>(h.width);

  if (h.format == ftkPixelFormat::GRAY8 || h.format == ftkPixelFormat::GRAY8_VIS ||
      h.format == ftkPixelFormat::GRAY8_SL) {
    check_cuda(cudaMemcpy2DAsync(tensor->pointer(),
                                 row_bytes,
                                 frame_->imageLeftPixels,
                                 h.imageStrideInBytes,
                                 row_bytes,
                                 h.height,
                                 cudaMemcpyHostToDevice,
                                 cuda_stream),
               "AtracsysMasterSourceOp: failed to upload 8-bit frame");
    return true;
  }

  if (h.format == ftkPixelFormat::GRAY16 || h.format == ftkPixelFormat::GRAY16_VIS ||
      h.format == ftkPixelFormat::GRAY16_SL) {
    HOLOSCAN_LOG_ERROR(
        "AtracsysMasterSourceOp: received unexpected 16-bit {} frame ({}x{}); this path expects "
        "8-bit output",
        frame_kind,
        h.width,
        h.height);
    return false;
  }

  HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: unsupported {} frame format {}",
                    frame_kind,
                    static_cast<int>(h.format));
  return false;
}

AtracsysMasterSourceOp::SchedulerMode AtracsysMasterSourceOp::configured_scheduler_mode() const {
  if (scheduler_mode_.get() == "exclusive") {
    return SchedulerMode::kExclusive;
  }
  if (scheduler_mode_.get() != "mixed") {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: unknown scheduler_mode '{}', defaulting to mixed",
                      scheduler_mode_.get());
  }
  return SchedulerMode::kMixed;
}

atracsys::HardwareMode AtracsysMasterSourceOp::configured_initial_hw_mode() const {
  if (initial_hw_mode_.get() == "ir" || initial_hw_mode_.get() == "infrared") {
    return atracsys::HardwareMode::kInfrared;
  }
  if (initial_hw_mode_.get() == "structured") {
    return atracsys::HardwareMode::kStructured;
  }
  if (initial_hw_mode_.get() != "visible") {
    HOLOSCAN_LOG_WARN("AtracsysMasterSourceOp: unknown initial_hw_mode '{}', defaulting to visible",
                      initial_hw_mode_.get());
  }
  return atracsys::HardwareMode::kVisible;
}

}  // namespace holoscan::ops
