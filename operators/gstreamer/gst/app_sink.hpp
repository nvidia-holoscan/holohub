/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN__GSTREAMER__GST__APP_SINK_HPP
#define HOLOSCAN__GSTREAMER__GST__APP_SINK_HPP

#include <gst/app/gstappsink.h>

#include "caps.hpp"
#include "element.hpp"
#include "sample.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Base template class for GstAppSink-derived wrappers
 */
template <typename Derived, typename NativeType>
class AppSinkBase : public ElementBase<Derived, NativeType> {
 public:
  explicit AppSinkBase(NativeType* appsink = nullptr)
      : ElementBase<Derived, NativeType>(appsink) {}
  explicit AppSinkBase(const ObjectBase<Derived, NativeType>& other)
      : ElementBase<Derived, NativeType>(other) {}
  explicit AppSinkBase(ObjectBase<Derived, NativeType>&& other)
      : ElementBase<Derived, NativeType>(std::move(other)) {}

  /// @brief Set callbacks for the AppSink element
  /// @param callbacks Pointer to ::GstAppSinkCallbacks structure
  /// @param user_data User data to pass to callbacks
  /// @param notify Destroy notify callback for user_data
  /// @note Wraps gst_app_sink_set_callbacks for type-safe usage
  void set_callbacks(::GstAppSinkCallbacks* callbacks, ::gpointer user_data,
                     ::GDestroyNotify notify) {
    gst_app_sink_set_callbacks(GST_APP_SINK(this->get()), callbacks, user_data, notify);
  }

  /// @brief Try to pull a sample from the appsink with timeout
  /// @param timeout Maximum time to wait (0 = don't wait, GST_CLOCK_TIME_NONE = wait forever)
  /// @return Sample if available, empty Sample otherwise
  /// @note Wraps gst_app_sink_try_pull_sample for type-safe usage
  Sample try_pull_sample(::GstClockTime timeout) {
    return Sample(gst_app_sink_try_pull_sample(GST_APP_SINK(this->get()), timeout));
  }
};

/**
 * @brief Wrapper class for ::GstAppSink
 */
class AppSink : public AppSinkBase<AppSink, ::GstAppSink> {
 public:
  explicit AppSink(::GstAppSink* appsink = nullptr) : AppSinkBase(appsink) {}
  explicit AppSink(const AppSinkBase& other) : AppSinkBase(other) {}
  explicit AppSink(AppSinkBase&& other) : AppSinkBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static ::GType get_type_func() { return GST_TYPE_APP_SINK; }
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__APP_SINK_HPP */
