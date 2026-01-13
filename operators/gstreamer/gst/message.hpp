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

#ifndef HOLOSCAN__GSTREAMER__GST__MESSAGE_HPP
#define HOLOSCAN__GSTREAMER__GST__MESSAGE_HPP

#include <gst/gst.h>

#include <string>

#include "error.hpp"
#include "mini_object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstMessage with automatic cleanup
 *
 * This class manages the lifetime of a GstMessage object and automatically
 * calls gst_message_unref when destroyed.
 */
class Message : public MiniObjectBase<Message, ::GstMessage> {
 public:
  /**
   * @brief Constructor from raw pointer (takes ownership)
   * @param message GstMessage pointer to wrap (nullptr is allowed)
   */
  explicit Message(::GstMessage* message = nullptr) : MiniObjectBase(message) {}

  /**
   * @brief Get the type of this message
   * @return The GstMessageType (e.g., GST_MESSAGE_ERROR, GST_MESSAGE_EOS)
   * @note Returns GST_MESSAGE_UNKNOWN if the message is null
   */
  GstMessageType get_type() const {
    return this->get() ? GST_MESSAGE_TYPE(this->get()) : GST_MESSAGE_UNKNOWN;
  }

  /**
   * @brief Parse error message and extract error information.
   * @return Error object containing the GError (automatically freed on destruction).
   * @throws std::runtime_error if the underlying GstMessage is null
   */
  Error parse_error() const;

  /**
   * @brief Parse error message and extract error and debug information.
   * @param debug_info Reference to string to receive debug info.
   *                   The debug info will be copied to this string.
   *                   The memory is automatically managed (no manual g_free needed).
   * @return Error object containing the GError (automatically freed on destruction).
   * @throws std::runtime_error if the underlying GstMessage is null
   */
  Error parse_error(std::string& debug_info) const;

  /**
   * @brief Parse state changed message and extract state information.
   * @param oldstate Pointer to GstState to receive the old state
   * @param newstate Pointer to GstState to receive the new state
   * @param pending Pointer to GstState to receive the pending state
   * @note All parameters can be nullptr if the information is not needed
   * @throws std::runtime_error if the underlying GstMessage is null
   */
  void parse_state_changed(GstState* oldstate, GstState* newstate, GstState* pending) const;

  static constexpr auto ref_func = gst_message_ref;
  static constexpr auto unref_func = gst_message_unref;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__MESSAGE_HPP */
