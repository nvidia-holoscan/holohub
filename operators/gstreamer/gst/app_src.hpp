/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN__GSTREAMER__GST__APP_SRC_HPP
#define HOLOSCAN__GSTREAMER__GST__APP_SRC_HPP

#include <gst/app/gstappsrc.h>

#include "buffer.hpp"
#include "caps.hpp"
#include "element.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Base template class for GstAppSrc-derived wrappers
 */
template <typename Derived, typename NativeType>
class AppSrcBase : public ElementBase<Derived, NativeType> {
 public:
  explicit AppSrcBase(NativeType* appsrc = nullptr) : ElementBase<Derived, NativeType>(appsrc) {}
  explicit AppSrcBase(const ObjectBase<Derived, NativeType>& other)
      : ElementBase<Derived, NativeType>(other) {}
  explicit AppSrcBase(ObjectBase<Derived, NativeType>&& other)
      : ElementBase<Derived, NativeType>(std::move(other)) {}

  /// @brief Push a buffer into the AppSrc element
  /// @param buffer The buffer to push (will be ref'd automatically)
  /// @returns GstFlowReturn indicating success or failure
  /// @note Wraps gst_app_src_push_buffer for type-safe usage
  ::GstFlowReturn push_buffer(Buffer buffer) {
    return gst_app_src_push_buffer(GST_APP_SRC(this->get()), buffer.ref().get());
  }

  /// @brief Signal end of stream to the AppSrc element
  /// @returns GstFlowReturn indicating success or failure
  /// @note Wraps gst_app_src_end_of_stream for type-safe usage
  ::GstFlowReturn end_of_stream() { return gst_app_src_end_of_stream(GST_APP_SRC(this->get())); }
};

/**
 * @brief Wrapper class for ::GstAppSrc
 */
class AppSrc : public AppSrcBase<AppSrc, ::GstAppSrc> {
 public:
  explicit AppSrc(::GstAppSrc* appsrc = nullptr) : AppSrcBase(appsrc) {}
  explicit AppSrc(const AppSrcBase& other) : AppSrcBase(other) {}
  explicit AppSrc(AppSrcBase&& other) : AppSrcBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static ::GType get_type_func() { return GST_TYPE_APP_SRC; }
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__APP_SRC_HPP */
