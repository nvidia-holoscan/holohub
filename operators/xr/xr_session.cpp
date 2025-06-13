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

#include "xr_session.hpp"

#include "xr_plugin.hpp"

#include <atomic>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

namespace holoscan {

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
}

void XrSession::register_plugin(std::reference_wrapper<IXrPlugin> plugin) {
  plugins_.push_back(plugin);
}

void XrSession::initialize() {
  if (is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Resource '{}' is already initialized. Skipping...", name());
    return;
  }

  Resource::initialize();

  // OpenXR extensions to enable.
  std::unordered_set<std::string_view> xr_extensions = {
      XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME,
      XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
      XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME,
  };

  for (auto& plugin : plugins_) {
    const auto plugin_extensions = plugin.get().get_required_instance_extensions();

    for (const auto ext_str : plugin_extensions) {
      xr_extensions.insert(ext_str);
    }
  }

  // Create an OpenXR instance.
  const std::vector<const char*> xr_api_layers = {};

  std::vector<const char*> out_ext = {};
  out_ext.reserve(xr_extensions.size());
  for (const std::string_view& sv : xr_extensions) {
    out_ext.push_back(sv.data());
  }

  xr_instance_ = xr::createInstanceUnique({
      {},
      {application_name_.get().c_str(),
       application_version_.get(),
       kEngineName,
       kEngineVersion,
       xr::Version::current()},
      static_cast<uint32_t>(xr_api_layers.size()),
      xr_api_layers.data(),
      static_cast<uint32_t>(out_ext.size()),
      out_ext.data(),
  });
  dispatch_ = xr::DispatchLoaderDynamic(*xr_instance_);
  xr::SystemId system_id = xr_instance_->getSystem({xr::FormFactor::HeadMountedDisplay});
  view_configurations_ = xr_instance_->enumerateViewConfigurationViewsToVector(
      system_id, xr::ViewConfigurationType::PrimaryStereo);
  blend_modes_ = xr_instance_->enumerateEnvironmentBlendModesToVector(
      system_id, xr::ViewConfigurationType::PrimaryStereo);

  // TODO: Enable XR_EXT_view_configuration_depth_range extension and read from
  // xrEnumerateViewConfigurationViews if supported.
  view_configuration_depth_range_.recommendedNearZ = 0.02f;
  view_configuration_depth_range_.minNearZ = 0.02f;
  view_configuration_depth_range_.recommendedFarZ = 100.f;
  view_configuration_depth_range_.maxFarZ = std::numeric_limits<float>::infinity();

  // Create a Vulkan instance using the XR_KHR_vulkan_enable2 extension.
  xr::GraphicsRequirementsVulkanKHR graphics_requirements_vulkan =
      xr_instance_->getVulkanGraphicsRequirements2KHR(system_id, dispatch_);
  vk::ApplicationInfo applicationInfo{
      .pApplicationName = application_name_.get().c_str(),
      .applicationVersion = application_version_.get(),
      .pEngineName = kEngineName,
      .engineVersion = kEngineVersion,
      .apiVersion = VK_API_VERSION_1_3,
  };
  VkInstance vk_instance;
  VkResult vk_result;
  vk::InstanceCreateInfo instance_create_info{
      .pApplicationInfo = &applicationInfo,
  };
  VkInstanceCreateInfo vk_instance_create_info =
      static_cast<VkInstanceCreateInfo>(instance_create_info);
  xr::VulkanInstanceCreateInfoKHR xr_vk_instance_create_info(
      system_id,
      {},
      vk_context_.getDispatcher()->vkGetInstanceProcAddr,
      &vk_instance_create_info,
      nullptr);
  xr_instance_->createVulkanInstanceKHR(
      xr_vk_instance_create_info, &vk_instance, &vk_result, dispatch_);
  if (vk_result != VK_SUCCESS) {
    throw std::runtime_error("xrCreateVulkanInstanceKHR failed");
  }
  vk_instance_ = vk::raii::Instance(vk_context_, vk_instance);

  // Create a Vulkan device and queue.
  VkPhysicalDevice vk_physical_device;
  xr_instance_->getVulkanGraphicsDevice2KHR(
      {system_id, vk_instance}, &vk_physical_device, dispatch_);
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
      system_id,
      {},
      vk_instance_.getDispatcher()->vkGetInstanceProcAddr,
      vk_physical_device,
      &vk_device_create_info,
      nullptr);
  VkDevice vk_device;
  xr_instance_->createVulkanDeviceKHR(xr_vk_device_create_info, &vk_device, &vk_result, dispatch_);
  if (vk_result != VK_SUCCESS) {
    throw std::runtime_error("xrCreateVulkanDeviceKHR failed");
  }
  uint32_t kQueueIndex = 0;
  vk_device_ = vk::raii::Device(vk_physical_device_, vk_device);
  vk_queue_ = vk_device_.getQueue(vk_queue_family_index_, kQueueIndex);

  // Create an OpenXR session with a Vulkan graphics binding.
  xr::GraphicsBindingVulkanKHR graphics_binding_vulkan(
      vk_instance, vk_physical_device, vk_device, vk_queue_family_index_, kQueueIndex);
  session_ = xr_instance_->createSessionUnique({{}, system_id, graphics_binding_vulkan.get()});

  reference_space_ = session_->createReferenceSpaceUnique(
      {xr::ReferenceSpaceType::Local, xr::Posef({0, 0, 0, 1}, {0, 0, 0})});
  view_space_ = session_->createReferenceSpaceUnique(
      {xr::ReferenceSpaceType::View, xr::Posef({0, 0, 0, 1}, {0, 0, 0})});

  // Begin the OpenXR session.
  session_->beginSession({xr::ViewConfigurationType::PrimaryStereo});

  // Wait for the session to become ready.
  while (session_state_ != xr::SessionState::Ready) {
    xr::EventDataBuffer event_data_buffer{};
    const xr::Result result = xr_instance_->pollEvent(event_data_buffer);
    if (result == xr::Result::EventUnavailable) {
      continue;
    }
    if (result != xr::Result::Success) {
      throw std::runtime_error("xrPollEvent failed");
    }
    if (event_data_buffer.type == xr::StructureType::EventDataSessionStateChanged) {
      const xr::EventDataSessionStateChanged& session_state_changed_event =
          reinterpret_cast<xr::EventDataSessionStateChanged&>(event_data_buffer);
      session_state_ = session_state_changed_event.state;
    }
  }

  for (auto& plugin : plugins_) {
    plugin.get().on_session_created();
  }

  // not safe to maintain ref to Resources after initializion
  plugins_.clear();

  HOLOSCAN_LOG_INFO("OpenXR session is ready");
  openxr_initialized_ = true;
}

XrSession::~XrSession() {
  session_->requestExitSession();
}

xr::SpaceLocation XrSession::locate_view_space() const noexcept(false) {
  if (!openxr_initialized_) {
    HOLOSCAN_LOG_WARN("XRSession is not initialized");
    return {};
  }

  timespec now_time;
  if (clock_gettime(CLOCK_MONOTONIC, &now_time) == -1) {
    HOLOSCAN_LOG_ERROR("clock_gettime return an error");
    return {};
  }

  const auto now_xr_time = xr_instance_->convertTimespecTimeToTimeKHR(&now_time, dispatch_);
  return view_space_->locateSpace(*reference_space_, now_xr_time);
}

}  // namespace holoscan
