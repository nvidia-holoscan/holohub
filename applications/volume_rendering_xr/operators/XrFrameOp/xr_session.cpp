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

#include "xr_session.hpp"

#include "holoscan/holoscan.hpp"
#define XR_EXT_HAND_INTERACTION_EXTENSION_NAME "XR_EXT_hand_interaction"
#include <thread>

namespace holoscan::openxr {

static constexpr char kEngineName[] = "Holoscan XR";
static constexpr uint32_t kEngineVersion = 1;

void XrSession::setup(ComponentSpec& spec) {
  spec.param<std::string>(application_name_,
                          "application_name",
                          "Application Name",
                          "OpenXR Application Name",
                          "Holoscan XR");
  spec.param(application_version_,
             "application_version",
             "Application Version",
             "OpenXR Application Version",
             0u);
  spec.param(near_z_, "near_z", "Near Z", "Distance to the near frustum plane", 0.1f);
  spec.param(far_z_, "far_z", "Far Z", "Distance to the far frustum plane", 100.f);
}

void XrSession::initialize() {
  // check if resource is already initialized
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Resource '{}' is already initialized. Skipping...", name());
    return;
  }

  Resource::initialize();

  // TODO: Remove after parameter initialization is fixed in Holoscan.
  auto& params = spec_->params();
  for (auto& arg : args_) {
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' not found in spec_->params()", arg.name());
      continue;
    }
    ArgumentSetter::set_param(params[arg.name()], arg);
  }

  // Gather required and desired (if supported) OpenXR extensions.
  std::vector<const char*> xr_extensions = {
      XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME,
      XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
  };
  const std::vector<const char*> xr_desired_extensions = {
      XR_ML_ML2_CONTROLLER_INTERACTION_EXTENSION_NAME,
      XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME,
      XR_EXT_HAND_INTERACTION_EXTENSION_NAME,
  };
  const std::vector<xr::ExtensionProperties> extension_properties =
      xr::enumerateInstanceExtensionPropertiesToVector(nullptr);
  for (const char* desired_extension : xr_desired_extensions) {
    for (const xr::ExtensionProperties& supported_extension : extension_properties) {
      if (std::strcmp(desired_extension, supported_extension.extensionName) == 0) {
        xr_extensions.push_back(supported_extension.extensionName);
      }
    }
  }
  for (const char* extension : xr_extensions) {
    xr_extensions_.emplace(extension);
  }

  // Create an OpenXR instance.
  const std::vector<const char*> xr_api_layers = {};
  xr_instance_ = xr::createInstanceUnique({
      {},
      {application_name_.get().c_str(),
       application_version_.get(),
       kEngineName,
       kEngineVersion,
       xr::Version::current()},
      static_cast<uint32_t>(xr_api_layers.size()),
      xr_api_layers.data(),
      static_cast<uint32_t>(xr_extensions.size()),
      xr_extensions.data(),
  });
  xr::DispatchLoaderDynamic dispatch{*xr_instance_};
  xr::SystemId system_id = xr_instance_->getSystem({xr::FormFactor::HeadMountedDisplay});
  view_configurations_ = xr_instance_->enumerateViewConfigurationViewsToVector(
      system_id, xr::ViewConfigurationType::PrimaryStereo);
  blend_modes_ = xr_instance_->enumerateEnvironmentBlendModesToVector(
      system_id, xr::ViewConfigurationType::PrimaryStereo);

  // TODO: Enable XR_EXT_view_configuration_depth_range extension and read from
  // xrEnumerateViewConfigurationViews if supported.
  view_configuration_depth_range_.recommendedNearZ = near_z_.get();
  view_configuration_depth_range_.minNearZ = near_z_.get();
  view_configuration_depth_range_.recommendedFarZ = far_z_.get();
  view_configuration_depth_range_.maxFarZ = far_z_.get();

  // Create Vulkan resources using the XR_KHR_vulkan_enable2 extension.
  xr::GraphicsRequirementsVulkanKHR graphics_requirements_vulkan =
      xr_instance_->getVulkanGraphicsRequirements2KHR(system_id, dispatch);
  vk::ApplicationInfo applicationInfo{
      .pApplicationName = application_name_.get().c_str(),
      .applicationVersion = application_version_.get(),
      .pEngineName = kEngineName,
      .engineVersion = kEngineVersion,
      .apiVersion = VK_API_VERSION_1_1,
  };
  VkInstance vk_instance;
  VkResult vk_result;
  vk::InstanceCreateInfo instance_create_info{
      .pApplicationInfo = &applicationInfo,
  };
  VkInstanceCreateInfo vk_instance_create_info =
      static_cast<VkInstanceCreateInfo>(instance_create_info);
  xr::VulkanInstanceCreateInfoKHR xr_vk_instance_create_info(
      system_id, {}, &vkGetInstanceProcAddr, &vk_instance_create_info, nullptr);
  xr_instance_->createVulkanInstanceKHR(
      xr_vk_instance_create_info, &vk_instance, &vk_result, dispatch);
  if (vk_result != VK_SUCCESS) {
    throw std::runtime_error("xrCreateVulkanInstanceKHR failed");
  }
  vk_instance_ = vk::raii::Instance(vk_context_, vk_instance);
  VkPhysicalDevice vk_physical_device;
  xr_instance_->getVulkanGraphicsDevice2KHR(
      {system_id, vk_instance}, &vk_physical_device, dispatch);
  vk_physical_device_ = vk::raii::PhysicalDevice(vk_instance_, vk_physical_device);
  std::vector<vk::QueueFamilyProperties> queue_family_props =
      vk_physical_device_.getQueueFamilyProperties();
  for (int i = 0; i < queue_family_props.size(); ++i) {
    if (queue_family_props[i].queueFlags &
        (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eTransfer)) {
      vk_queue_family_index_ = i;
      break;
    }
  }
  float queue_priority = 0.0f;
  vk::DeviceQueueCreateInfo device_queue_create_info({
      .queueFamilyIndex = vk_queue_family_index_,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
  });
  const std::vector<const char*> vk_device_extensions = {
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
  };
  vk::DeviceCreateInfo device_create_info({
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &device_queue_create_info,
      .enabledExtensionCount = static_cast<uint32_t>(vk_device_extensions.size()),
      .ppEnabledExtensionNames = vk_device_extensions.data(),
  });
  VkDeviceCreateInfo vk_device_create_info = static_cast<VkDeviceCreateInfo>(device_create_info);
  xr::VulkanDeviceCreateInfoKHR xr_vk_device_create_info(
      system_id, {}, &vkGetInstanceProcAddr, vk_physical_device, &vk_device_create_info, nullptr);
  VkDevice vk_device;
  xr_instance_->createVulkanDeviceKHR(xr_vk_device_create_info, &vk_device, &vk_result, dispatch);
  if (vk_result != VK_SUCCESS) {
    throw std::runtime_error("xrCreateVulkanDeviceKHR failed");
  }
  uint32_t kQueueIndex = 0;
  vk_device_ = vk::raii::Device(vk_physical_device_, vk_device);
  vk_queue_ = vk_device_.getQueue(vk_queue_family_index_, kQueueIndex);

  // Create a cuda stream for synchronizing CUDA events with the vulkan device queue.
  if (cudaStreamCreate(&cuda_stream_) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }

  // Create and begin an OpenXR session with a Vulkan graphics binding.
  xr::GraphicsBindingVulkanKHR graphics_binding_vulkan(
      vk_instance, vk_physical_device, vk_device, vk_queue_family_index_, kQueueIndex);
  session_ = xr_instance_->createSessionUnique({{}, system_id, graphics_binding_vulkan.get()});
  reference_space_ = session_->createReferenceSpaceUnique(
      {xr::ReferenceSpaceType::Local, xr::Posef({0, 0, 0, 1}, {0, 0, 0})});
  view_space_ = session_->createReferenceSpaceUnique(
      {xr::ReferenceSpaceType::View, xr::Posef({0, 0, 0, 1}, {0, 0, 0})});

  using namespace std::literals::chrono_literals;
  // transitioning to xr::SessionState::Ready will invoke beginSession.
  if (!transition_state(xr::SessionState::Ready, 3000ms)) {
    throw std::runtime_error(
        "Failed to begin session, transitioning to ready session state timed out.");
  }

  HOLOSCAN_LOG_DEBUG("OpenXR session begun");
}

XrSession::~XrSession() {
  session_->requestExitSession();
  cudaStreamDestroy(cuda_stream_);
}

// Return event if one is available, otherwise return null.
const XrEventDataBaseHeader* XrSession::try_read_next_event(xr::EventDataBuffer& eventDataBuffer) {
  // It is sufficient to clear the just the XrEventDataBuffer header to
  // XR_TYPE_EVENT_DATA_BUFFER
  XrEventDataBaseHeader* baseHeader =
      reinterpret_cast<XrEventDataBaseHeader*>(eventDataBuffer.get());
  *baseHeader = {XR_TYPE_EVENT_DATA_BUFFER};
  const xr::Result xr = xr_instance_->pollEvent(eventDataBuffer);
  if (xr == XR_SUCCESS) {
    if (baseHeader->type == XR_TYPE_EVENT_DATA_EVENTS_LOST) {
      const XrEventDataEventsLost* const eventsLost =
          reinterpret_cast<const XrEventDataEventsLost*>(baseHeader);
      HOLOSCAN_LOG_DEBUG("%d events lost", eventsLost->lostEventCount);
    }
    return baseHeader;
  }
  if (xr == XR_EVENT_UNAVAILABLE) {
    return nullptr;
  }
  throw std::runtime_error(std::string("xrPollEvent Failed!, error code: ") +
                           std::to_string((std::int32_t)xr));
}

bool XrSession::transition_state(const xr::SessionState new_state,
                                 const std::chrono::milliseconds timeout_ms) {
  using namespace std::literals::chrono_literals;
  const auto start = std::chrono::steady_clock::now();
  while (session_state_ != new_state) {
    const auto pollResults = poll_events();
    if (pollResults.exit_render_loop || pollResults.request_reestart) {
      return false;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (elapsed >= timeout_ms) {
      break;
    }
    std::this_thread::sleep_for(10ms);
  }
  return session_state_ == new_state;
}

XrSession::PollResult XrSession::poll_events() {
  PollResult pollResult{};

  // Process all pending messages.
  while (const XrEventDataBaseHeader* event = try_read_next_event(eventDataBuffer_)) {
    switch (event->type) {
      case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
        const auto& instanceLossPending =
            *reinterpret_cast<const XrEventDataInstanceLossPending*>(event);
        HOLOSCAN_LOG_WARN("XrEventDataInstanceLossPending by %lld", instanceLossPending.lossTime);
        pollResult = {.exit_render_loop = true, .request_reestart = true};
        return pollResult;
      }
      case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
        auto sessionStateChangedEvent =
            *reinterpret_cast<const XrEventDataSessionStateChanged*>(event);
        process_session_state_changed(sessionStateChangedEvent, pollResult);
        break;
      }
      case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
        HOLOSCAN_LOG_DEBUG("Interaction profile changed.");
        break;
      case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
      default: {
        HOLOSCAN_LOG_DEBUG("Ignoring event type %d", static_cast<int>(event->type));
        break;
      }
    }
  }

  return pollResult;
}

void XrSession::process_session_state_changed(
    const XrEventDataSessionStateChanged& stateChangedEvent, XrSession::PollResult& pollState) {
  const xr::SessionState oldState = session_state_;
  session_state_ = static_cast<xr::SessionState>(stateChangedEvent.state);

  if ((stateChangedEvent.session != XR_NULL_HANDLE) &&
      (stateChangedEvent.session != session_.getRawHandle())) {
    HOLOSCAN_LOG_ERROR("XrEventDataSessionStateChanged for unknown session");
    return;
  }

  switch (session_state_) {
    case xr::SessionState::Ready: {
      session_->beginSession({xr::ViewConfigurationType::PrimaryStereo});
      break;
    }
    case xr::SessionState::Stopping: {
      session_->endSession();
      break;
    }
    case xr::SessionState::Exiting: {
      pollState = {
          .exit_render_loop = true,
          // Do not attempt to restart because user closed this session.
          .request_reestart = false,
      };
      break;
    }
    case xr::SessionState::LossPending: {
      pollState = {
          .exit_render_loop = true,
          // Poll for a new instance.
          .request_reestart = true,
      };
      break;
    }
    default:
      break;
  }
}

}  // namespace holoscan::openxr
