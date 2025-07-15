/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef XR_HAND_TRACKER_HPP
#define XR_HAND_TRACKER_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#include "openxr/openxr.h"
#include "openxr/openxr_platform.h"

#include "xr_session.hpp"

#include "openxr/openxr.hpp"

#include "holoscan/holoscan.hpp"
#include "xr_plugin.hpp"

namespace holoscan {

class XrHandTracker : public holoscan::Resource, public IXrPlugin {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(XrHandTracker)

  XrHandTracker(std::shared_ptr<holoscan::XrSession>, xr::HandEXT hand);
  ~XrHandTracker() override = default;

  [[nodiscard]] std::vector<std::string_view> get_required_instance_extensions() override;
  void on_session_created() noexcept(false) override;

  [[nodiscard]] std::optional<std::vector<xr::HandJointLocationEXT>> locate_hand_joints() noexcept(
      false);

 private:
  xr::HandEXT hand_;
  std::atomic<xr::HandTrackerEXT> hand_tracker_handle_;
};

}  // namespace holoscan

#endif
