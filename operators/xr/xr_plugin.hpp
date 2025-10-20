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

#ifndef XR_PLUGIN_HPP
#define XR_PLUGIN_HPP

#include <chrono>
#include <memory>
#include <set>
#include <string_view>
#include <vector>

#include "vulkan/vulkan.h"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_raii.hpp"

#include "openxr/openxr.h"
#include "openxr/openxr_platform.h"

#include "xr_session.hpp"

#include "openxr/openxr.hpp"

namespace holoscan {

class IXrPlugin {
 public:
  explicit IXrPlugin(std::shared_ptr<holoscan::XrSession> xr_session) noexcept(false)
      : xr_session_(xr_session) {
    if (!xr_session) {
      throw std::runtime_error("invalid xr_session pointer (nullptr)");
    }
    xr_session->register_plugin(*this);
  }
  virtual ~IXrPlugin() = default;
  [[nodiscard]] virtual std::vector<std::string_view> get_required_instance_extensions() = 0;
  // signals OpenXR handles have been populated in the XrSession obj.
  virtual void on_session_created() = 0;

 protected:
  std::shared_ptr<XrSession> xr_session_{};
};

}  // namespace holoscan

#endif
