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

#ifndef XR_SESSION_HPP
#define XR_SESSION_HPP

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#include "openxr/openxr.h"
#include "openxr/openxr_platform.h"

#include "openxr/openxr.hpp"

#include "holoscan/holoscan.hpp"

namespace holoscan {

class IXrPlugin;

// Manages an OpenXR session.
class XrSession : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(XrSession)

  void setup(ComponentSpec& spec) override;
  void initialize() override;
  void register_plugin(std::reference_wrapper<IXrPlugin> plugin);
  xr::SpaceLocation locate_view_space() const noexcept(false);

  ~XrSession();

  // Returns the OpenXR session.
  xr::Session& get() { return *session_; }

  xr::Instance& instance() { return *xr_instance_; }
  xr::DispatchLoaderDynamic& dispatch() { return dispatch_; }
  xr::Space& reference_space() { return *reference_space_; }
  xr::Space& view_space() { return *view_space_; }
  const std::vector<xr::ViewConfigurationView>& view_configurations() {
    return view_configurations_;
  }
  xr::ViewConfigurationDepthRangeEXT view_configuration_depth_range() {
    return view_configuration_depth_range_;
  }
  xr::EnvironmentBlendMode blend_mode() { return blend_modes_[0]; }


  vk::raii::PhysicalDevice& vk_physical_device() { return vk_physical_device_; }
  vk::raii::Device& vk_device() { return vk_device_; }
  uint32_t vk_queue_family_index() { return vk_queue_family_index_; }
  vk::raii::Queue& vk_queue() { return vk_queue_; }

 private:
  Parameter<std::string> application_name_;
  Parameter<uint32_t> application_version_;

  xr::UniqueInstance xr_instance_;
  xr::UniqueSession session_;
  xr::UniqueSpace reference_space_;
  xr::UniqueSpace view_space_;
  xr::SessionState session_state_{xr::SessionState::Unknown};
  std::vector<xr::ViewConfigurationView> view_configurations_;
  xr::ViewConfigurationDepthRangeEXT view_configuration_depth_range_;
  std::vector<xr::EnvironmentBlendMode> blend_modes_;
  mutable xr::DispatchLoaderDynamic dispatch_;
  std::atomic_bool openxr_initialized_{false};

  vk::raii::Context vk_context_;
  vk::raii::Instance vk_instance_{nullptr};
  vk::raii::PhysicalDevice vk_physical_device_{nullptr};
  vk::raii::Device vk_device_{nullptr};
  uint32_t vk_queue_family_index_;
  vk::raii::Queue vk_queue_{nullptr};
  std::vector<std::reference_wrapper<IXrPlugin>> plugins_{};
};

}  // namespace holoscan

#endif /* XR_SESSION_HPP */
