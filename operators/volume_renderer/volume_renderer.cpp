/* SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "volume_renderer.hpp"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <gxf/core/entity.hpp>
#include <gxf/cuda/cuda_event.hpp>
#include <gxf/multimedia/camera.hpp>
#include <gxf/multimedia/video.hpp>

#include <ClaraVizRenderer.h>

#include <claraviz/interface/CameraInterface.h>
#include <claraviz/interface/DataInterface.h>
#include <claraviz/interface/DataViewInterface.h>
#include <claraviz/interface/ImageInterface.h>
#include <claraviz/interface/JsonInterface.h>
#include <claraviz/interface/LightInterface.h>
#include <claraviz/interface/PostProcessDenoiseInterface.h>
#include <claraviz/interface/PostProcessTonemapInterface.h>
#include <claraviz/interface/RenderSettingsInterface.h>
#include <claraviz/interface/TransferFunctionInterface.h>
#include <claraviz/interface/ViewInterface.h>

#include "dataset.hpp"
#include "video_buffer_blob.hpp"

#include <chrono>
#include <memory>
#include <vector>

static clara::viz::Matrix4x4 toMatrix(const nvidia::gxf::Pose3D& pose) {
  return clara::viz::Matrix4x4(
      {{{{pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.translation[0]}},
        {{pose.rotation[3], pose.rotation[4], pose.rotation[5], pose.translation[1]}},
        {{pose.rotation[6], pose.rotation[7], pose.rotation[8], pose.translation[2]}},
        {{0.f, 0.f, 0.f, 1.f}}}});
}

static clara::viz::Vector2f toTangentX(const nvidia::gxf::CameraModel& camera_model) {
  return clara::viz::Vector2f(
      -camera_model.principal_point.x / camera_model.focal_length.x,
      (camera_model.dimensions.x - camera_model.principal_point.x) / camera_model.focal_length.x);
}

static clara::viz::Vector2f toTangentY(const nvidia::gxf::CameraModel& camera_model) {
  return clara::viz::Vector2f(
      camera_model.principal_point.y / camera_model.focal_length.y,
      -(camera_model.dimensions.y - camera_model.principal_point.y) / camera_model.focal_length.y);
}

namespace holoscan::ops {

/**
 * This class provides the functionality to start rendering and wait for the rendered image to be
 * returned.
 **/
class ImageService : public clara::viz::MessageReceiver, public clara::viz::MessageProvider {
 public:
  ImageService() = default;
  virtual ~ImageService() {}

  /**
   * Render a image.
   */
  void Render(uint32_t width, uint32_t height,
              const std::shared_ptr<VideoBufferBlob>& color_buffer_blob,
              const std::shared_ptr<VideoBufferBlob>& depth_buffer_blob) {
    auto image_message = std::make_shared<clara::viz::ImageMessage>();

    image_message->view_name_ = "";
    image_message->width_ = width;
    image_message->height_ = height;
    image_message->color_type_ = clara::viz::ColorImageType::RAW_RGBA_U8;
    image_message->color_memory_ = color_buffer_blob;
    image_message->depth_type_ = clara::viz::DepthImageType::RAW_DEPTH_F32;
    image_message->depth_memory_ = depth_buffer_blob;

    SendMessage(image_message);
  }

  /**
   * Wait for the rendered image to be returned by the renderer and return it.
   *
   * @returns the rendered image
   **/
  std::shared_ptr<const clara::viz::ImageEncodeMessage> WaitForRenderedImage() {
#if 1
    // @todo the first iteration needs to compile the cuda shaders and takes more than a second,
    // handle this
    Wait();
#else
    const Status status = WaitFor(std::chrono::milliseconds(1000));
    if (status == Status::TIMEOUT) {
      throw InvalidState() << "Timeout occurred while waiting for rendered image";
    }
#endif
    auto message = DequeueMessage();
    if (message->GetID() == clara::viz::ImageEncodeMessage::id_) {
      return std::static_pointer_cast<const clara::viz::ImageEncodeMessage>(message);
    } else if (message->GetID() == clara::viz::ImageRenderFailedMessage::id_) {
      throw InvalidState()
          << "Renderer fail with "
          << std::static_pointer_cast<const clara::viz::ImageRenderFailedMessage>(message)->reason_;
    } else {
      throw InvalidState() << "Unexpected message: " << message->GetID().GetName();
    }
  }
};

struct VolumeRendererOp::Impl {
  Parameter<std::vector<IOSpec*>> settings_;
  Parameter<std::vector<IOSpec*>> merge_settings_;
  Parameter<std::string> config_file_;
  Parameter<std::string> write_config_file_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<uint32_t> alloc_width_;
  Parameter<uint32_t> alloc_height_;

  cudaStream_t stream_ = nullptr;
  std::vector<clara::viz::Vector2f> limits_;

  std::shared_ptr<ImageService> image_service_;

  std::unique_ptr<clara::viz::Renderer> renderer_;

  clara::viz::CameraInterface camera_interface_;
  clara::viz::CameraApertureInterface camera_aperture_interface_;

  std::shared_ptr<clara::viz::DataInterface> data_interface_;
  clara::viz::DataConfigInterface data_config_interface_;
  std::shared_ptr<clara::viz::DataHistogramInterface> data_histogram_interface_;

  clara::viz::DataCropInterface data_crop_interface_;
  clara::viz::DataTransformInterface data_transform_interface_;
  clara::viz::DataViewInterface data_view_interface_;
  clara::viz::BackgroundLightInterface background_light_interface_;
  clara::viz::LightInterface light_interface_;
  clara::viz::PostProcessDenoiseInterface post_process_denoise_interface_;
  clara::viz::PostProcessTonemapInterface post_process_tonemap_interface_;
  clara::viz::RenderSettingsInterface render_settings_interface_;
  clara::viz::TransferFunctionInterface transfer_function_interface_;
  clara::viz::ViewInterface view_interface_;

  std::unique_ptr<clara::viz::JsonInterface> json_interface_;

  Dataset dataset_;
  uint32_t current_dataset_frame_ = 0;
  std::chrono::steady_clock::time_point last_dataset_frame_start_;
  /// If 0 then foveated rendering if off, else it's a counter initialized when a valid gaze
  /// direction is received and counted down when no valid gaze direction is received. Foveated
  /// rendering is wwitched off when the counter reaches 0. This is done to avoid frequent on/off
  /// switches with the eye blink of the user.
  uint32_t eye_gaze_counter_ = 0;
  /// foveation defaults
  bool default_enable_foveation_ = false;
  float default_warp_full_resolution_size_ = 1.f;
  float default_warp_resolution_scale_ = 1.f;
};

bool receive_volume(InputContext& input, Dataset& dataset, Dataset::Types type) {
  std::string name(type == Dataset::Types::Density ? "density" : "mask");

  auto volume = input.receive<holoscan::gxf::Entity>((name + "_volume").c_str());
  auto spacing_input = input.receive<std::array<float, 3>>((name + "_spacing").c_str());
  auto permute_axis_input =
      input.receive<std::array<uint32_t, 3>>((name + "_permute_axis").c_str());
  auto flip_axes_input = input.receive<std::array<bool, 3>>((name + "_flip_axes").c_str());

  if (volume) {
    nvidia::gxf::Handle<nvidia::gxf::Tensor> volume_tensor =
        static_cast<nvidia::gxf::Entity>(volume.value()).get<nvidia::gxf::Tensor>("volume").value();

    std::array<float, 3> spacing{1.f, 1.f, 1.f};
    std::array<uint32_t, 3> permute_axis{0, 1, 2};
    std::array<bool, 3> flip_axes{false, false, false};

    if (spacing_input) { spacing = *spacing_input; }
    if (permute_axis_input) { permute_axis = *permute_axis_input; }
    if (flip_axes_input) { flip_axes = *flip_axes_input; }

    dataset.SetVolume(type, spacing, permute_axis, flip_axes, volume_tensor);

    return true;
  }

  return false;
}

void VolumeRendererOp::initialize() {
  const std::vector<uint32_t> cuda_device_ordinals{0};

  clara::viz::LogLevel log_level;
  switch (holoscan::log_level()) {
    case holoscan::LogLevel::TRACE:
    case holoscan::LogLevel::DEBUG:
      log_level = clara::viz::LogLevel::Debug;
      break;
    case holoscan::LogLevel::INFO:
      log_level = clara::viz::LogLevel::Info;
      break;
    case holoscan::LogLevel::WARN:
      log_level = clara::viz::LogLevel::Warning;
      break;
    case holoscan::LogLevel::ERROR:
    case holoscan::LogLevel::CRITICAL:
    case holoscan::LogLevel::OFF:
      log_level = clara::viz::LogLevel::Error;
      break;
    default:
      throw std::runtime_error("Unhandled Holoscan log level");
  }

  // create the image service
  impl_->image_service_.reset(new ImageService);

  // create and start the volume renderer
  impl_->renderer_ = std::make_unique<clara::viz::Renderer>(
      std::shared_ptr<clara::viz::MessageReceiver>(),
      std::static_pointer_cast<clara::viz::MessageReceiver>(impl_->image_service_),
      cuda_device_ordinals,
      clara::viz::VolumeRenderBackend::Default,
      log_level);

  // create the interfaces, these will generate messages which are then handled by the renderer
  const std::shared_ptr<clara::viz::MessageReceiver>& receiver = impl_->renderer_->GetReceiver();

  impl_->camera_interface_.RegisterReceiver(receiver);
  impl_->camera_aperture_interface_.RegisterReceiver(receiver);

  impl_->data_interface_ = std::make_shared<clara::viz::DataInterface>();
  impl_->data_interface_->RegisterReceiver(receiver);
  impl_->data_config_interface_.RegisterReceiver(receiver);
  // the data interface needs to get updates from the data config interface to do proper parameter
  // validation
  impl_->data_config_interface_.RegisterReceiver(impl_->data_interface_);

  impl_->data_histogram_interface_ = std::make_shared<clara::viz::DataHistogramInterface>();
  impl_->data_histogram_interface_->RegisterReceiver(receiver);

  impl_->data_crop_interface_.RegisterReceiver(receiver);
  impl_->data_transform_interface_.RegisterReceiver(receiver);
  impl_->data_view_interface_.RegisterReceiver(receiver);
  impl_->background_light_interface_.RegisterReceiver(receiver);
  impl_->light_interface_.RegisterReceiver(receiver);
  impl_->post_process_denoise_interface_.RegisterReceiver(receiver);
  impl_->post_process_tonemap_interface_.RegisterReceiver(receiver);
  impl_->render_settings_interface_.RegisterReceiver(receiver);
  impl_->transfer_function_interface_.RegisterReceiver(receiver);
  impl_->view_interface_.RegisterReceiver(receiver);

  // renderer will also receive messages from the image service
  impl_->image_service_->RegisterReceiver(receiver);

  // start the renderer thread
  impl_->renderer_->Run();

  // init json settings
  impl_->json_interface_ =
      std::make_unique<clara::viz::JsonInterface>(&impl_->background_light_interface_,
                                                  &impl_->camera_interface_,
                                                  &impl_->camera_aperture_interface_,
                                                  &impl_->data_config_interface_,
                                                  impl_->data_histogram_interface_.get(),
                                                  &impl_->data_crop_interface_,
                                                  &impl_->data_transform_interface_,
                                                  &impl_->data_view_interface_,
                                                  &impl_->light_interface_,
                                                  &impl_->post_process_denoise_interface_,
                                                  &impl_->post_process_tonemap_interface_,
                                                  &impl_->render_settings_interface_,
                                                  &impl_->transfer_function_interface_,
                                                  &impl_->view_interface_);
  impl_->json_interface_->InitSettings();

  // call base class
  Operator::initialize();
}

void VolumeRendererOp::start() {
  if (cudaStreamCreate(&impl_->stream_) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }

  if (!impl_->config_file_.get().empty()) {
    // setup renderer by reading settings from the configuration file
    std::ifstream input_file_stream(impl_->config_file_.get());
    if (!input_file_stream) {
      throw std::runtime_error("Could not open configuration " + impl_->config_file_.get() +
                               " for reading");
    }

    const nlohmann::json settings = nlohmann::json::parse(input_file_stream);

    // set the configuration from the settings
    impl_->json_interface_->MergeSettings(settings);

    if (settings.contains("dataset")) {
      auto& dataset_settings = settings.at("dataset");
      impl_->dataset_.SetFrameDuration(std::chrono::duration<float, std::chrono::seconds::period>(
          dataset_settings.value("frameDuration", 1.f)));
    }
  }
}

void VolumeRendererOp::stop() {
  if (impl_->stream_) {
    if (cudaStreamDestroy(impl_->stream_) != cudaSuccess) {
      throw std::runtime_error("cudaStreamDestroy failed");
    }
  }
}

void VolumeRendererOp::setup(OperatorSpec& spec) {
  impl_.reset(new Impl);

  spec.param(impl_->settings_,
             "settings",
             "Settings",
             "Vector of JSON settings inputs passed to clara::viz::JsonInterface::SetSettings() on "
             "compute()",
             {});
  spec.param(impl_->merge_settings_,
             "merge_settings",
             "Merge Settings",
             "Vector of JSON settings inputs passed to clara::viz::JsonInterface::MergeSettings() "
             "on compute() (after calling 'SetSettings')",
             {});
  spec.param(impl_->config_file_,
             "config_file",
             "Configuration file",
             "Name of the JSON renderer configuration file to load",
             std::string(""));
  spec.param(impl_->write_config_file_,
             "write_config_file",
             "Write config settings file",
             "Deduce config settings from volume data and write to file. Sets a light in correct "
             "distance. Sets a transfer function using the histogram of the data. Writes the "
             "JSON configuration to the file with the given name",
             std::string(""));
  spec.param(impl_->allocator_,
             "allocator",
             "Allocator",
             "Allocator used to allocate render buffer outputs.");
  spec.param(impl_->alloc_width_,
             "alloc_width",
             "Alloc Width",
             "Width of the render buffer to allocate when no pre-allocated buffers are provided.",
             1024u);
  spec.param(impl_->alloc_height_,
             "alloc_height",
             "Alloc Height",
             "Height of the render buffer to allocate when no pre-allocated buffers are provided.",
             768u);

  spec.input<nvidia::gxf::Pose3D>("volume_pose").condition(ConditionType::kNone);
  spec.input<std::array<nvidia::gxf::Vector2f, 3>>("crop_box").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Vector2f>("depth_range").condition(ConditionType::kNone);

  spec.input<nvidia::gxf::Pose3D>("left_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::Pose3D>("right_camera_pose").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("left_camera_model").condition(ConditionType::kNone);
  spec.input<nvidia::gxf::CameraModel>("right_camera_model").condition(ConditionType::kNone);

  spec.input<nvidia::gxf::Pose3D>("eye_gaze_pose").condition(ConditionType::kNone);

  spec.input<std::array<float, 16>>("camera_matrix").condition(ConditionType::kNone);

  spec.input<holoscan::gxf::Entity>("color_buffer_in").condition(ConditionType::kNone);
  spec.input<holoscan::gxf::Entity>("depth_buffer_in").condition(ConditionType::kNone);

  spec.input<holoscan::gxf::Entity>("density_volume").condition(ConditionType::kNone);
  spec.input<std::array<float, 3>>("density_spacing").condition(ConditionType::kNone);
  spec.input<std::array<uint32_t, 3>>("density_permute_axis").condition(ConditionType::kNone);
  spec.input<std::array<bool, 3>>("density_flip_axes").condition(ConditionType::kNone);

  spec.input<holoscan::gxf::Entity>("mask_volume").condition(ConditionType::kNone);
  spec.input<std::array<float, 3>>("mask_spacing").condition(ConditionType::kNone);
  spec.input<std::array<uint32_t, 3>>("mask_permute_axis").condition(ConditionType::kNone);
  spec.input<std::array<bool, 3>>("mask_flip_axes").condition(ConditionType::kNone);

  spec.output<holoscan::gxf::Entity>("color_buffer_out");
  spec.output<holoscan::gxf::Entity>("depth_buffer_out").condition(ConditionType::kNone);
}

void VolumeRendererOp::compute(InputContext& input, OutputContext& output,
                               ExecutionContext& context) {
  // get the density volumes
  bool new_volume = receive_volume(input, impl_->dataset_, Dataset::Types::Density);
  if (!receive_volume(input, impl_->dataset_, Dataset::Types::Segmentation)) {
    // there are datasets without segmentation volume, if we receive a density volume
    // only, reset the segmentation volume
    impl_->dataset_.ResetVolume(Dataset::Types::Segmentation);
  } else {
    new_volume = true;
  }
  if (new_volume) {
    impl_->dataset_.Configure(impl_->data_config_interface_);
    impl_->dataset_.Set(*impl_->data_interface_.get());

    // the volume is defined, if the config file is empty we can deduce settings now
    if (impl_->config_file_.get().empty()) {
      impl_->json_interface_->DeduceSettings(clara::viz::ViewMode::CINEMATIC);
    }

    if (!impl_->write_config_file_.get().empty()) {
      // get the settings and write to file
      const nlohmann::json settings = impl_->json_interface_->GetSettings();
      std::ofstream output_file_stream(impl_->write_config_file_.get());
      if (!output_file_stream) {
        throw std::runtime_error("Could not open configuration for writing");
      }
      output_file_stream << settings;
    }

    {
      clara::viz::DataCropInterface::AccessGuard access(impl_->data_crop_interface_);
      impl_->limits_ = access->limits.Get();
      if (impl_->limits_.empty()) {
        impl_->limits_ = {clara::viz::Vector2f(0.f, 1.f),
                          clara::viz::Vector2f(0.f, 1.f),
                          clara::viz::Vector2f(0.f, 1.f),
                          clara::viz::Vector2f(0.f, 1.f)};
      }
    }
    {
      clara::viz::RenderSettingsInterface::AccessGuard access(impl_->render_settings_interface_);
      impl_->default_enable_foveation_ = access->enable_foveation;
      impl_->default_warp_resolution_scale_ = access->warp_resolution_scale.Get();
      impl_->default_warp_full_resolution_size_ = access->warp_full_resolution_size.Get();
    }
  }

  // get the input buffers
  auto color_message = input.receive<holoscan::gxf::Entity>("color_buffer_in");
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> color_buffer;
  if (color_message) {
    color_buffer = static_cast<nvidia::gxf::Entity&>(color_message.value())
                       .get<nvidia::gxf::VideoBuffer>()
                       .value();
  } else {
    color_message = holoscan::gxf::Entity::New(&context);
    if (!color_message) { throw std::runtime_error("Failed to allocate entity; terminating."); }
    color_buffer = static_cast<nvidia::gxf::Entity&>(color_message.value())
                       .add<nvidia::gxf::VideoBuffer>()
                       .value();

    // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), impl_->allocator_->gxf_cid());

    color_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
        impl_->alloc_width_,
        impl_->alloc_height_,
        nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR,
        nvidia::gxf::MemoryStorageType::kDevice,
        allocator.value());
    if (!color_buffer->pointer()) {
      throw std::runtime_error("Failed to allocate color render buffer.");
    }
  }

  auto depth_message = input.receive<holoscan::gxf::Entity>("depth_buffer_in");
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> depth_buffer;
  if (depth_message) {
    depth_buffer = static_cast<nvidia::gxf::Entity&>(depth_message.value())
                       .get<nvidia::gxf::VideoBuffer>()
                       .value();
  }

  // apply new JSON settings
  auto settings = input.receive<std::vector<nlohmann::json>>("settings");
  auto merge_settings = input.receive<std::vector<nlohmann::json>>("merge_settings");
  if (settings) {
    for (const auto& setting : settings.value()) { impl_->json_interface_->SetSettings(setting); }
  }
  if (merge_settings) {
    for (const auto& setting : merge_settings.value()) {
      impl_->json_interface_->MergeSettings(setting);
    }
  }

  // update cameras
  {
    clara::viz::CameraInterface::AccessGuard access(impl_->camera_interface_);
    auto camera = access->GetCamera();

    auto left_pose = input.receive<nvidia::gxf::Pose3D>("left_camera_pose");
    if (left_pose) { camera->left_eye_pose = toMatrix(*left_pose); }
    auto left_model = input.receive<nvidia::gxf::CameraModel>("left_camera_model");
    if (left_model) {
      camera->left_tangent_x = toTangentX(*left_model);
      camera->left_tangent_y = toTangentY(*left_model);
    }

    auto right_pose = input.receive<nvidia::gxf::Pose3D>("right_camera_pose");
    if (right_pose) { camera->right_eye_pose = toMatrix(*right_pose); }
    auto right_model = input.receive<nvidia::gxf::CameraModel>("right_camera_model");
    if (right_model) {
      camera->right_tangent_x = toTangentX(*right_model);
      camera->right_tangent_y = toTangentY(*right_model);
    }

    auto eye_gaze_pose = input.receive<nvidia::gxf::Pose3D>("eye_gaze_pose");
    if (eye_gaze_pose) {
      if (!impl_->eye_gaze_counter_) {
        // switch on
        clara::viz::RenderSettingsInterface::AccessGuard access(impl_->render_settings_interface_);
        access->enable_foveation = true;
        access->warp_full_resolution_size.Set(.4f);
        access->warp_resolution_scale.Set(.28f);
      }
      // initialize the counter when we have a valid gaze pose, after this number of frames
      // without a valid gaze pose ar received foveated rendering is disabled.
      constexpr uint32_t EYE_GAZE_SWITCH_OFF_FRAMES = 10;
      impl_->eye_gaze_counter_ = EYE_GAZE_SWITCH_OFF_FRAMES;

      // there are some unknown discrepancies between the direction ClaraViz expects and the
      // direction OpenXR provides, needed to negate x to make it work correctly.
      clara::viz::Vector3f direction{-eye_gaze_pose->rotation.at(6),
                                     eye_gaze_pose->rotation.at(7),
                                     eye_gaze_pose->rotation.at(8)};

      camera->left_gaze_direction.Set(direction);
      camera->right_gaze_direction.Set(direction);
    }
    if (impl_->eye_gaze_counter_ && !bool(eye_gaze_pose)) {
      // count down
      --impl_->eye_gaze_counter_;
      if (!impl_->eye_gaze_counter_) {
        // switch off;
        clara::viz::Vector3f direction{0.f, 0.f, 1.f};
        camera->left_gaze_direction.Set(direction);
        camera->right_gaze_direction.Set(direction);

        clara::viz::RenderSettingsInterface::AccessGuard access(impl_->render_settings_interface_);
        access->enable_foveation = impl_->default_enable_foveation_;
        access->warp_full_resolution_size.Set(impl_->default_warp_full_resolution_size_);
        access->warp_resolution_scale.Set(impl_->default_warp_resolution_scale_);
      }
    }

    auto camera_matrix = input.receive<std::shared_ptr<std::array<float, 16>>>("camera_matrix");
    if (camera_matrix && camera_matrix.value()) {
      // convert the camera matrix to eye, up and look_at vectors
      clara::viz::Vector3f eye(camera_matrix.value()->at(3 + 0 * 4),
                               camera_matrix.value()->at(3 + 1 * 4),
                               camera_matrix.value()->at(3 + 2 * 4));
      // we can't yet initialize the camera used by Holoviz, therefore we have to adapt the vectors
      // a bit
      eye(0) *= 4.f;
      eye(1) *= 4.f;
      eye(2) *= 4.f;
      camera->eye.Set(eye);

      clara::viz::Vector3f up(camera_matrix.value()->at(1 + 0 * 4),
                              camera_matrix.value()->at(1 + 1 * 4),
                              camera_matrix.value()->at(1 + 2 * 4));
      // normalize
      const float inv_length = 1.f / std::sqrt(up(0) * up(0) + up(1) * up(1) + up(2) * up(2));
      up(0) *= inv_length;
      up(1) *= inv_length;
      up(2) *= inv_length;
      up(1) = -up(1);
      camera->up.Set(up);

      clara::viz::Vector3f look_at(camera_matrix.value()->at(2 + 0 * 4),
                                   camera_matrix.value()->at(2 + 1 * 4),
                                   camera_matrix.value()->at(2 + 2 * 4));
      camera->look_at.Set(look_at);
    }

    auto depth_range = input.receive<nvidia::gxf::Vector2f>("depth_range");
    if (depth_range) {
      clara::viz::Vector2f range(depth_range->x, depth_range->y);
      camera->depth_range.Set(range);
      camera->depth_clip.Set(range);
    }
  }

  // set volume transform matrix
  auto volume_pose = input.receive<nvidia::gxf::Pose3D>("volume_pose");
  if (volume_pose) {
    clara::viz::DataTransformInterface::AccessGuard access(impl_->data_transform_interface_);
    access->matrix = toMatrix(volume_pose.value());
  }

  // set volume cropping limits
  auto crop_limits = input.receive<std::array<nvidia::gxf::Vector2f, 3>>("crop_box");
  if (crop_limits) {
    std::array<nvidia::gxf::Vector2f, 3>& crop = crop_limits.value();
    std::vector<clara::viz::Vector2f> limits = impl_->limits_;
    for (int i = 1; i < 4; i++) {
      limits[i] = clara::viz::Vector2f(
          std::max(impl_->limits_[i](0), std::min(impl_->limits_[i](1), crop[i - 1].x)),
          std::min(impl_->limits_[i](1), std::max(impl_->limits_[i](0), crop[i - 1].y)));
      // ensure there is a delta between the planes
      if (limits[i](1) == limits[i](0)) {
        if (limits[i](1) + std::numeric_limits<float>::epsilon() > impl_->limits_[i](1)) {
          // decrease lower limit
          limits[i](0) -= std::numeric_limits<float>::epsilon();
        } else {
          // increase upper limit
          limits[i](1) += std::numeric_limits<float>::epsilon();
        }
      }
    }

    clara::viz::DataCropInterface::AccessGuard access(impl_->data_crop_interface_);
    access->limits.Set(limits);
  }

  // update the dataset frame number for animated datasets
  if (impl_->dataset_.GetNumberFrames() > 1) {
    if (impl_->last_dataset_frame_start_ == std::chrono::steady_clock::time_point()) {
      impl_->last_dataset_frame_start_ = std::chrono::steady_clock::now();
    }

    bool dataset_frame_changed = false;
    while (std::chrono::steady_clock::now() - impl_->last_dataset_frame_start_ >=
           impl_->dataset_.GetFrameDuration()) {
      dataset_frame_changed = true;
      impl_->current_dataset_frame_++;
      if (impl_->current_dataset_frame_ >= impl_->dataset_.GetNumberFrames()) {
        impl_->current_dataset_frame_ = 0;
      }

      impl_->last_dataset_frame_start_ +=
          std::chrono::duration_cast<std::chrono::milliseconds>(impl_->dataset_.GetFrameDuration());
    }
    if (dataset_frame_changed) {
      // switch to the new dataset
      impl_->dataset_.Set(*impl_->data_interface_.get(), impl_->current_dataset_frame_);
    }
  }

  // render
  const std::shared_ptr<VideoBufferBlob> color_buffer_blob(new VideoBufferBlob(color_buffer));
  std::shared_ptr<VideoBufferBlob> depth_buffer_blob;
  if (depth_buffer) { depth_buffer_blob = std::make_shared<VideoBufferBlob>(depth_buffer); }
  impl_->image_service_->Render(color_buffer->video_frame_info().width,
                                color_buffer->video_frame_info().height,
                                color_buffer_blob,
                                depth_buffer_blob);

  // then wait for the message with the encoded data
  const auto image = impl_->image_service_->WaitForRenderedImage();

  {
    color_buffer_blob->AccessConst(impl_->stream_);

    nvidia::gxf::Handle<nvidia::gxf::CudaEvent> cuda_event =
        static_cast<nvidia::gxf::Entity&>(color_message.value())
            .add<nvidia::gxf::CudaEvent>()
            .value();
    cuda_event->init();
    if (cudaEventRecord(cuda_event->event().value(), impl_->stream_) != cudaSuccess) {
      throw std::runtime_error("cudaEventRecord failed");
    }
    output.emit(color_message.value(), "color_buffer_out");
  }

  if (depth_buffer_blob) {
    depth_buffer_blob->AccessConst(impl_->stream_);

    nvidia::gxf::Handle<nvidia::gxf::CudaEvent> cuda_event =
        static_cast<nvidia::gxf::Entity&>(depth_message.value())
            .add<nvidia::gxf::CudaEvent>()
            .value();
    cuda_event->init();
    if (cudaEventRecord(cuda_event->event().value(), impl_->stream_) != cudaSuccess) {
      throw std::runtime_error("cudaEventRecord failed");
    }
    output.emit(depth_message.value(), "depth_buffer_out");
  }
}

}  // namespace holoscan::ops
