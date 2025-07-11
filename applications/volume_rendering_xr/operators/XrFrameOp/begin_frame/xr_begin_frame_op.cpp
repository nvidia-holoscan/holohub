/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "xr_begin_frame_op.hpp"

#include "Eigen/Dense"
#include "gxf/core/entity.hpp"
#include "gxf/cuda/cuda_event.hpp"
#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"

namespace holoscan::openxr {

void XrBeginFrameOp::setup(OperatorSpec& spec) {
  spec.output<XrFrame>("xr_frame");

  // camera state
  spec.output<nvidia::gxf::Pose3D>("left_camera_pose").condition(ConditionType::kNone);
  spec.output<nvidia::gxf::Pose3D>("right_camera_pose").condition(ConditionType::kNone);
  spec.output<nvidia::gxf::CameraModel>("left_camera_model").condition(ConditionType::kNone);
  spec.output<nvidia::gxf::CameraModel>("right_camera_model").condition(ConditionType::kNone);
  spec.output<nvidia::gxf::Vector2f>("depth_range").condition(ConditionType::kNone);

  // input state
  spec.output<bool>("trigger_click").condition(ConditionType::kNone);
  spec.output<bool>("shoulder_click").condition(ConditionType::kNone);
  spec.output<bool>("trackpad_touch").condition(ConditionType::kNone);
  spec.output<std::array<float, 2>>("trackpad").condition(ConditionType::kNone);

  // aim pose in local space
  spec.output<nvidia::gxf::Pose3D>("aim_pose").condition(ConditionType::kNone);
  // grip pose in local space
  spec.output<nvidia::gxf::Pose3D>("grip_pose").condition(ConditionType::kNone);
  // head pose in local space
  spec.output<nvidia::gxf::Pose3D>("head_pose").condition(ConditionType::kNone);
  // eye gaze pose in view space
  spec.output<nvidia::gxf::Pose3D>("eye_gaze_pose").condition(ConditionType::kNone);

  // render buffer
  spec.output<holoscan::gxf::Entity>("color_buffer");
  spec.output<holoscan::gxf::Entity>("depth_buffer");

  spec.param(session_, "session", "OpenXR Session", "OpenXR and Vulkan context");
  spec.param(enable_eye_tracking_,
             "enable_eye_tracking",
             "Eye tracking enable switch",
             "Enable eye tracking");
}

void XrBeginFrameOp::start() {
  std::shared_ptr<holoscan::openxr::XrSession> session = session_.get();

  uint32_t view_width = session->display_width();
  uint32_t view_height = session->display_height();

  // Create swapchains for the projection layer (color, depth).
  // NOTE: The two stereo views of the projection layer are _vertically stacked_
  // for ease of use when renderer.
  color_swapchain_ = XrCudaInteropSwapchain::create(*session,
                                                    {
                                                        {},
                                                        xr::SwapchainUsageFlagBits::TransferDst,
                                                        VK_FORMAT_R8G8B8A8_SRGB,
                                                        session->swapchain_count(),
                                                        view_width,
                                                        view_height * 2,
                                                        /*faceCount=*/1,
                                                        /*arraySize=*/1,
                                                        /*mipCount=*/1,
                                                    });
  depth_swapchain_ =
      XrCudaInteropSwapchain::create(*session,
                                     {
                                         {},
                                         xr::SwapchainUsageFlagBits::TransferDst |
                                             xr::SwapchainUsageFlagBits::DepthStencilAttachment,
                                         VK_FORMAT_D32_SFLOAT,
                                         session->swapchain_count(),
                                         view_width,
                                         view_height * 2,
                                         /*faceCount=*/1,
                                         /*arraySize=*/1,
                                         /*mipCount=*/1,
                                     });

  // Create action sets for controller
  xr::Instance& xr_instance = session->xr_instance();
  xr::Session& xr_session = session->handle();

  action_set_ = xr_instance.createActionSetUnique({"control", "Controller Actions", 0});

  action_map_["trigger_click"] = action_set_->createActionUnique(
      {"click", xr::ActionType::BooleanInput, 0, nullptr, "trigger click"});
  action_map_["shoulder_click"] = action_set_->createActionUnique(
      {"shoulder_click", xr::ActionType::BooleanInput, 0, nullptr, "shoulder click"});
  action_map_["trackpad_touch"] = action_set_->createActionUnique(
      {"trackpad_touch", xr::ActionType::BooleanInput, 0, nullptr, "trackpad touch"});
  action_map_["trackpad_x"] = action_set_->createActionUnique(
      {"trackpad_x", xr::ActionType::FloatInput, 0, nullptr, "trackpad x"});
  action_map_["trackpad_y"] = action_set_->createActionUnique(
      {"trackpad_y", xr::ActionType::FloatInput, 0, nullptr, "trackpad y"});
  action_map_["aim_pose"] = action_set_->createActionUnique(
      {"aim_pose", xr::ActionType::PoseInput, 0, nullptr, "aim pose"});
  action_map_["grip_pose"] = action_set_->createActionUnique(
      {"grip_pose", xr::ActionType::PoseInput, 0, nullptr, "grip pose"});

  // check if eye tracking is enabled
  if (enable_eye_tracking_) {
    // check if the eye gaze extension is supported
    if (session->xr_extensions().count(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME)) {
      // check for eye gaze support
      xr::SystemEyeGazeInteractionPropertiesEXT eye_gaze_properties;
      xr::SystemProperties system_properties(&eye_gaze_properties);
      xr::SystemId system_id = xr_instance.getSystem({xr::FormFactor::HeadMountedDisplay});
      xr_instance.getSystemProperties(system_id, system_properties);
      if (eye_gaze_properties.supportsEyeGazeInteraction) {
        action_map_["eye_gaze"] = action_set_->createActionUnique(
            {"gaze_action", xr::ActionType::PoseInput, 0, nullptr, "Gaze Action"});

        std::vector<xr::ActionSuggestedBinding> bindings_eye_gaze = {
            {*action_map_["eye_gaze"],
             xr_instance.stringToPath("/user/eyes_ext/input/gaze_ext/pose")},
        };

        xr::InteractionProfileSuggestedBinding suggested_bindings_simple(
            xr_instance.stringToPath("/interaction_profiles/ext/eye_gaze_interaction"),
            bindings_eye_gaze.size(),
            bindings_eye_gaze.data());
        xr_instance.suggestInteractionProfileBindings(suggested_bindings_simple);

        eye_gaze_space_.reset(xr_session.createActionSpace(
            {*action_map_["eye_gaze"], xr::Path::null(), xr::Posef({0, 0, 0, 1}, {0, 0, 0})}));

        holoscan::log_info("Eye tracking is active");
      } else {
        holoscan::log_warn(
            "Eye tracking is enabled but the system is not supporting eye gaze interaction.");
      }
    } else {
      holoscan::log_warn(
          "Eye tracking is enabled but the extension XR_EXT_eye_gaze_interaction is not "
          "supported.");
    }
  }

  // Suggest bindings for the simple controller profile.
  std::vector<xr::ActionSuggestedBinding> bindings_simple = {
      {
          *action_map_["trigger_click"],
          xr_instance.stringToPath("/user/hand/left/input/select/click"),
      },
      {
          *action_map_["trigger_click"],
          xr_instance.stringToPath("/user/hand/right/input/select/click"),
      },
      {
          *action_map_["aim_pose"],
          xr_instance.stringToPath("/user/hand/left/input/aim/pose"),
      },
      {
          *action_map_["aim_pose"],
          xr_instance.stringToPath("/user/hand/right/input/aim/pose"),
      },
      {
          *action_map_["grip_pose"],
          xr_instance.stringToPath("/user/hand/left/input/grip/pose"),
      },
      {
          *action_map_["grip_pose"],
          xr_instance.stringToPath("/user/hand/right/input/grip/pose"),
      },
  };
  xr::InteractionProfileSuggestedBinding suggested_bindings_simple(
      xr_instance.stringToPath("/interaction_profiles/khr/simple_controller"),
      bindings_simple.size(),
      bindings_simple.data());
  xr_instance.suggestInteractionProfileBindings(suggested_bindings_simple);

  // Suggest bindings for the Magic Leap 2 controller profile
  if (session->xr_extensions().count(XR_ML_ML2_CONTROLLER_INTERACTION_EXTENSION_NAME)) {
    std::vector<xr::ActionSuggestedBinding> bindings_ml2 = {
        {
            *action_map_["trigger_click"],
            xr_instance.stringToPath("/user/hand/left/input/trigger/click"),
        },
        {
            *action_map_["trigger_click"],
            xr_instance.stringToPath("/user/hand/right/input/trigger/click"),
        },
        {
            *action_map_["shoulder_click"],
            xr_instance.stringToPath("/user/hand/left/input/shoulder/click"),
        },
        {
            *action_map_["shoulder_click"],
            xr_instance.stringToPath("/user/hand/right/input/shoulder/click"),
        },
        {
            *action_map_["trackpad_touch"],
            xr_instance.stringToPath("/user/hand/left/input/trackpad/touch"),
        },
        {
            *action_map_["trackpad_touch"],
            xr_instance.stringToPath("/user/hand/right/input/trackpad/touch"),
        },
        {
            *action_map_["trackpad_x"],
            xr_instance.stringToPath("/user/hand/left/input/trackpad/x"),
        },
        {
            *action_map_["trackpad_x"],
            xr_instance.stringToPath("/user/hand/right/input/trackpad/x"),
        },
        {
            *action_map_["trackpad_y"],
            xr_instance.stringToPath("/user/hand/left/input/trackpad/y"),
        },
        {
            *action_map_["trackpad_y"],
            xr_instance.stringToPath("/user/hand/right/input/trackpad/y"),
        },
        {
            *action_map_["aim_pose"],
            xr_instance.stringToPath("/user/hand/left/input/aim/pose"),
        },
        {
            *action_map_["aim_pose"],
            xr_instance.stringToPath("/user/hand/right/input/aim/pose"),
        },
        {
            *action_map_["grip_pose"],
            xr_instance.stringToPath("/user/hand/left/input/grip/pose"),
        },
        {
            *action_map_["grip_pose"],
            xr_instance.stringToPath("/user/hand/right/input/grip/pose"),
        },
    };
    xr::InteractionProfileSuggestedBinding suggested_bindings_ml2(
        xr_instance.stringToPath("/interaction_profiles/ml/ml2_controller"),
        bindings_ml2.size(),
        bindings_ml2.data());
    xr_instance.suggestInteractionProfileBindings(suggested_bindings_ml2);
  }

  // Suggest bindings for hand interaction profile
  if (session->xr_extensions().count("XR_EXT_hand_interaction")) {
    std::vector<xr::ActionSuggestedBinding> bindings_hands = {
        {
            *action_map_["trigger_click"],
            xr_instance.stringToPath("/user/hand/right/input/aim_activate_ext/value"),
        },
        {
            *action_map_["shoulder_click"],
            xr_instance.stringToPath("/user/hand/left/input/aim_activate_ext/value"),
        },
        {
            *action_map_["aim_pose"],
            xr_instance.stringToPath("/user/hand/right/input/pinch_ext/pose"),
        },
    };
    xr::InteractionProfileSuggestedBinding suggested_bindings_hands(
        xr_instance.stringToPath("/interaction_profiles/ext/hand_interaction_ext"),
        bindings_hands.size(),
        bindings_hands.data());
    xr_instance.suggestInteractionProfileBindings(suggested_bindings_hands);
  }

  std::array<xr::Path, 2> hand_paths{
      xr_instance.stringToPath("/user/hand/left"),
      xr_instance.stringToPath("/user/hand/right"),
  };

  // Attach the action set to the session.
  xr::SessionActionSetsAttachInfo attach_info(1, &*action_set_);
  xr_session.attachSessionActionSets(attach_info);
  for (std::size_t hand_idx = 0; hand_idx < hand_paths.size(); ++hand_idx) {
    aim_space_[hand_idx].reset(xr_session.createActionSpace(
        {*action_map_["aim_pose"], hand_paths[hand_idx], xr::Posef({0, 0, 0, 1}, {0, 0, 0})}));
    grip_space_[hand_idx].reset(xr_session.createActionSpace(
        {*action_map_["grip_pose"], hand_paths[hand_idx], xr::Posef({0, 0, 0, 1}, {0, 0, 0})}));
  }
}

void XrBeginFrameOp::compute(InputContext& input, OutputContext& output,
                             ExecutionContext& context) {
  XrFrame frame{
      .color_swapchain = *color_swapchain_,
      .depth_swapchain = *depth_swapchain_,
  };

  std::shared_ptr<holoscan::openxr::XrSession> session = session_.get();

  const auto pollResult = session->poll_events();
  // TODO: This can be handled more gracefully, checking pollResult.request_restart
  //       and re-creating the session/instance.
  if (pollResult.exit_render_loop) {
    throw std::runtime_error(
        "XrInstance/Session is or pending to be lost, or is currently exiting");
  }

  frame.state = session->handle().waitFrame({});
  xr::ViewLocateInfo view_locate_info(xr::ViewConfigurationType::PrimaryStereo,
                                      frame.state.predictedDisplayTime,
                                      session->reference_space());
  frame.views = session->handle().locateViewsToVector(view_locate_info, frame.view_state.put());
  session->handle().beginFrame({});
  output.emit(frame, "xr_frame");

  // emit camera intrinsics/extrinsics
  const nvidia::gxf::Pose3D left_camera_pose = toPose3D(frame.views[0]);
  const nvidia::gxf::Pose3D right_camera_pose = toPose3D(frame.views[1]);
  output.emit(left_camera_pose, "left_camera_pose");
  output.emit(right_camera_pose, "right_camera_pose");

  const nvidia::gxf::CameraModel left_camera_model =
      toCameraModel(frame.views[0], session->display_width(), session->display_height());
  const nvidia::gxf::CameraModel right_camera_model =
      toCameraModel(frame.views[1], session->display_width(), session->display_height());
  output.emit(left_camera_model, "left_camera_model");
  output.emit(right_camera_model, "right_camera_model");

  // emit clipping planes
  nvidia::gxf::Vector2f depth_range;
  depth_range.x = session->view_configuration_depth_range().recommendedNearZ;
  depth_range.y = session->view_configuration_depth_range().recommendedFarZ;
  output.emit(depth_range, "depth_range");

  // emit image buffers
  auto color_message = holoscan::gxf::Entity::New(&context);
  color_swapchain_->acquire(color_message);
  output.emit(color_message, "color_buffer");
  auto depth_message = holoscan::gxf::Entity::New(&context);
  depth_swapchain_->acquire(depth_message);
  output.emit(depth_message, "depth_buffer");

  // emit controller state
  xr::Session& xr_session = session->handle();
  xr::ActiveActionSet active_action_set(*action_set_, xr::Path::null());
  xr::ActionsSyncInfo actions_sync_info(1, &active_action_set);
  xr_session.syncActions(actions_sync_info);

  auto trigger_click = boolAction(xr_session, "trigger_click");
  output.emit(trigger_click, "trigger_click");
  auto shoulder_click = boolAction(xr_session, "shoulder_click");
  output.emit(shoulder_click, "shoulder_click");
  auto trackpad_touch = boolAction(xr_session, "trackpad_touch");
  output.emit(trackpad_touch, "trackpad_touch");
  std::array<float, 2> trackpad{
      xr_session.getActionStateFloat({*action_map_["trackpad_x"], xr::Path::null()}).currentState,
      xr_session.getActionStateFloat({*action_map_["trackpad_y"], xr::Path::null()}).currentState,
  };
  output.emit(trackpad, "trackpad");

  bool grip_pose_emitted = false;
  bool aim_pose_emitted = false;
  xr::Space space = session->reference_space();
  for (std::size_t hand_idx = 0; hand_idx < aim_space_.size(); ++hand_idx) {
    const auto aim_location =
        aim_space_[hand_idx]->locateSpace(space, frame.state.predictedDisplayTime);
    if (aim_location.locationFlags != xr::SpaceLocationFlagBits::None) {
      const xr::Posef& aim_pose = aim_location.pose;
      auto aim_pose_message = poseAction(aim_pose);
      output.emit(aim_pose_message, "aim_pose");
      aim_pose_emitted = true;
    }

    const auto grip_location =
        grip_space_[hand_idx]->locateSpace(space, frame.state.predictedDisplayTime);
    if (grip_location.locationFlags != xr::SpaceLocationFlagBits::None) {
      const xr::Posef& grip_pose = grip_location.pose;
      auto grip_pose_message = poseAction(grip_pose);
      output.emit(grip_pose_message, "grip_pose");
      grip_pose_emitted = true;
    }
  }

  static const xr::Posef id_pose{xr::Quaternionf{0, 0, 0, 1}, xr::Vector3f{0, 0, 0}};
  if (!aim_pose_emitted) { output.emit(poseAction(id_pose), "aim_pose"); }
  if (!grip_pose_emitted) { output.emit(poseAction(id_pose), "grip_pose"); }

  // emit device state
  xr::Posef head_pose =
      session->view_space().locateSpace(space, frame.state.predictedDisplayTime).pose;
  auto head_pose_message = poseAction(head_pose);
  output.emit(head_pose_message, "head_pose");

  if (eye_gaze_space_) {
    xr::SpaceLocation space_location =
        eye_gaze_space_->locateSpace(session->view_space(), frame.state.predictedDisplayTime);
    if ((space_location.locationFlags & xr::SpaceLocationFlagBits::PositionValid) &&
        (space_location.locationFlags & xr::SpaceLocationFlagBits::OrientationValid)) {
      auto eye_gaze_message = poseAction(space_location.pose);
      output.emit(eye_gaze_message, "eye_gaze_pose");
    }
  }
}

bool XrBeginFrameOp::boolAction(const xr::Session& xr_session, const std::string& name) {
  xr::ActionStateBoolean value =
      xr_session.getActionStateBoolean({*action_map_[name], xr::Path::null()});
  return bool(value.currentState);
}

nvidia::gxf::Pose3D XrBeginFrameOp::poseAction(const xr::Posef& pose) {
  nvidia::gxf::Pose3D pose3d;
  pose3d.translation[0] = pose.position.x;
  pose3d.translation[1] = pose.position.y;
  pose3d.translation[2] = pose.position.z;
  Eigen::Quaternionf orientation(
      pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(pose3d.rotation.data()) =
      orientation.matrix();

  return pose3d;
}

nvidia::gxf::Pose3D XrBeginFrameOp::toPose3D(const xr::View& view) {
  nvidia::gxf::Pose3D pose;
  pose.translation = {
      view.pose.position.x,
      view.pose.position.y,
      view.pose.position.z,
  };
  Eigen::Quaternionf orientation(view.pose.orientation.w,
                                 view.pose.orientation.x,
                                 view.pose.orientation.y,
                                 view.pose.orientation.z);
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(pose.rotation.data()) =
      orientation.matrix();
  return pose;
}

nvidia::gxf::CameraModel XrBeginFrameOp::toCameraModel(const xr::View& view, uint32_t display_width,
                                                       uint32_t display_height) {
  float left_tan = std::tan(-view.fov.angleLeft);
  float right_tan = std::tan(view.fov.angleRight);
  float up_tan = std::tan(view.fov.angleUp);
  float down_tan = std::tan(-view.fov.angleDown);

  nvidia::gxf::CameraModel camera;
  camera.dimensions = {display_width, display_height};
  camera.distortion_type = nvidia::gxf::DistortionType::Perspective;
  camera.focal_length.x = display_width / (left_tan + right_tan);
  camera.focal_length.y = display_height / (up_tan + down_tan);
  camera.principal_point.x = camera.focal_length.x * left_tan;
  camera.principal_point.y = camera.focal_length.y * up_tan;
  return camera;
}

}  // namespace holoscan::openxr
