/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef HOLOSCAN_OPERATORS_OPENXR_XR_SESSION_HPP
#define HOLOSCAN_OPERATORS_OPENXR_XR_SESSION_HPP

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr.hpp>

#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "holoscan/holoscan.hpp"
#include "xr_cuda_interop_swapchain.hpp"

namespace holoscan::openxr {

// Initializes and stores OpenXR session resources.
class XrSession : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(XrSession)

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  ~XrSession();

  xr::Session& handle() { return *session_; }
  xr::Instance& xr_instance() { return *xr_instance_; }
  xr::Space& reference_space() { return *reference_space_; }
  xr::Space& view_space() { return *view_space_; }
  xr::ViewConfigurationDepthRangeEXT view_configuration_depth_range() {
    return view_configuration_depth_range_;
  }
  const std::set<std::string> xr_extensions() { return xr_extensions_; }

  vk::raii::PhysicalDevice& vk_physical_device() { return vk_physical_device_; }
  vk::raii::Device& vk_device() { return vk_device_; }
  uint32_t vk_queue_family_index() { return vk_queue_family_index_; }
  vk::raii::Queue& vk_queue() { return vk_queue_; }

  cudaStream_t& cuda_stream() { return cuda_stream_; }

  uint32_t display_width() { return view_configurations_[0].recommendedImageRectWidth; }
  uint32_t display_height() { return view_configurations_[0].recommendedImageRectHeight; }
  uint32_t swapchain_count() { return view_configurations_[0].recommendedSwapchainSampleCount; }
  xr::EnvironmentBlendMode blend_mode() { return blend_modes_[0]; }

  struct PollResult {
    bool exit_render_loop = false;
    bool request_reestart = false;
  };
  PollResult poll_events();

  bool transition_state(const xr::SessionState new_state,
                        const std::chrono::milliseconds timeout_ms);

  bool is_session_running() const { return session_state_ == xr::SessionState::Ready; }

 private:
  const XrEventDataBaseHeader* try_read_next_event(xr::EventDataBuffer& eventDataBuffer);
  void process_session_state_changed(const XrEventDataSessionStateChanged& stateChangedEvent,
                                     PollResult& sessionState);

  Parameter<std::string> application_name_;
  Parameter<uint32_t> application_version_;
  Parameter<float> near_z_;
  Parameter<float> far_z_;

  xr::UniqueInstance xr_instance_;
  xr::UniqueSession session_;
  xr::UniqueSpace reference_space_;
  xr::UniqueSpace view_space_;
  std::vector<xr::ViewConfigurationView> view_configurations_;
  xr::ViewConfigurationDepthRangeEXT view_configuration_depth_range_;
  std::vector<xr::EnvironmentBlendMode> blend_modes_;
  std::set<std::string> xr_extensions_;
  xr::EventDataBuffer eventDataBuffer_{};
  xr::SessionState session_state_{xr::SessionState::Unknown};

  vk::raii::Context vk_context_;
  vk::raii::Instance vk_instance_{nullptr};
  vk::raii::PhysicalDevice vk_physical_device_{nullptr};
  vk::raii::Device vk_device_{nullptr};
  uint32_t vk_queue_family_index_;
  vk::raii::Queue vk_queue_{nullptr};

  cudaStream_t cuda_stream_;
};

// Metadata for an OpenXR frame.
struct XrFrame {
  xr::FrameState state;
  xr::ViewState view_state;
  std::vector<xr::View> views;

  XrCudaInteropSwapchain& color_swapchain;
  XrCudaInteropSwapchain& depth_swapchain;
};

}  // namespace holoscan::openxr

#endif /* HOLOSCAN_OPERATORS_OPENXR_XR_SESSION_HPP */
