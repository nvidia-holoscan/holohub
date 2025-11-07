/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN__GSTREAMER__GST__MESSAGE_HPP
#define HOLOSCAN__GSTREAMER__GST__MESSAGE_HPP

#include <gst/gst.h>

#include "object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstMessage with automatic cleanup
 *
 * This class manages the lifetime of a GstMessage object and automatically
 * calls gst_message_unref when destroyed.
 */
class Message : public Object<::GstMessage, gst_mini_object_ref_typed<::GstMessage>,
                              gst_mini_object_unref_typed<::GstMessage>> {
 public:
  /**
   * @brief Constructor from raw pointer (takes ownership)
   * @param message GstMessage pointer to wrap (nullptr is allowed)
   */
  explicit Message(::GstMessage* message = nullptr) : Object(message) {}
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__MESSAGE_HPP */
