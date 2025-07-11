/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "xr_hand_tracker.hpp"

#include <atomic>
#include <ctime>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace holoscan {

XrHandTracker::XrHandTracker(std::shared_ptr<holoscan::XrSession> xr_session, xr::HandEXT hand)
    : IXrPlugin(xr_session), hand_(hand) {}

std::vector<std::string_view> XrHandTracker::get_required_instance_extensions() {
  return {XR_EXT_HAND_TRACKING_EXTENSION_NAME, XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME};
}

void XrHandTracker::on_session_created() {
  const auto create_info =
      xr::HandTrackerCreateInfoEXT{hand_, xr::HandJointSetEXT{XR_HAND_JOINT_SET_DEFAULT_EXT}};
  hand_tracker_handle_ =
      xr_session_->get().createHandTrackerEXT(create_info, xr_session_->dispatch());
}

std::optional<std::vector<xr::HandJointLocationEXT>> XrHandTracker::locate_hand_joints() {
  auto hand_tracker_handle = hand_tracker_handle_.load(std::memory_order_relaxed);
  if (!hand_tracker_handle) {
    return {};
  }

  auto xr_instance = xr_session_->instance();
  auto dispatch = xr_session_->dispatch();

  timespec now_time;
  if (clock_gettime(CLOCK_MONOTONIC, &now_time) == -1) {
    HOLOSCAN_LOG_ERROR("clock_gettime return an error");
    return {};
  }

  const auto now_xr_time = xr_instance.convertTimespecTimeToTimeKHR(&now_time, dispatch);

  // Not using OpenXR-HPP here since the C++ structs do not set jointCount properly resulting in
  // validation error.
  XrHandJointsLocateInfoEXT locate_info{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};
  locate_info.baseSpace = xr_session_->reference_space().get();
  locate_info.time = now_xr_time.get();

  XrHandJointLocationsEXT joint_locations{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};
  joint_locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
  std::vector<xr::HandJointLocationEXT> joint_data(XR_HAND_JOINT_COUNT_EXT);
  joint_locations.jointLocations = reinterpret_cast<XrHandJointLocationEXT*>(joint_data.data());

  XrResult result =
      dispatch.xrLocateHandJointsEXT(hand_tracker_handle.get(), &locate_info, &joint_locations);
  if (result != XR_SUCCESS) {
    char buffer[XR_MAX_RESULT_STRING_SIZE];
    xrResultToString(xr_instance.get(), result, buffer);
    HOLOSCAN_LOG_ERROR("Failed to locate hand joints, error: %s", buffer);
    return {};
  }

  if (!joint_locations.isActive) {
    return {};
  }

  return joint_data;
}

}  // namespace holoscan
