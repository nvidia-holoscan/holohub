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

#ifndef HOLOSCAN__GSTREAMER__GST__BUS_HPP
#define HOLOSCAN__GSTREAMER__GST__BUS_HPP

#include <gst/gst.h>

#include "message.hpp"
#include "object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Templated base class for GStreamer bus types (GstBus and its derivatives)
 * @tparam Derived The derived class type (for CRTP pattern)
 * @tparam NativeType The GStreamer bus type (e.g. ::GstBus, specialized buses)
 *
 * This template provides common bus functionality for message handling and monitoring.
 * Derive concrete classes like Bus from this base class.
 */
template <typename Derived, typename NativeType>
class BusBase : public ObjectBase<Derived, NativeType> {
 public:
  explicit BusBase(NativeType* bus = nullptr) : ObjectBase<Derived, NativeType>(bus) {}

  // Implicit conversion from base ObjectBase type
  explicit BusBase(const ObjectBase<Derived, NativeType>& other)
      : ObjectBase<Derived, NativeType>(other) {}
  explicit BusBase(ObjectBase<Derived, NativeType>&& other)
      : ObjectBase<Derived, NativeType>(std::move(other)) {}

  /// @brief Pop a message from the bus with timeout and filtering
  /// @param timeout Timeout in nanoseconds (use GST_MSECOND, GST_SECOND macros)
  /// @param types Bitwise OR of GstMessageType values to filter for
  /// @returns A Message wrapper for the popped message (may be empty if timeout/no match)
  Message timed_pop_filtered(GstClockTime timeout, GstMessageType types) const {
    return Message(gst_bus_timed_pop_filtered(GST_BUS(this->get()), timeout, types));
  }

  // TODO: Add more common bus functionality here (post_message, poll, etc.)
};

/**
 * @brief Wrapper class for ::GstBus
 */
class Bus : public BusBase<Bus, ::GstBus> {
 public:
  explicit Bus(::GstBus* bus = nullptr) : BusBase(bus) {}

  // Implicit conversion from base BusBase type
  explicit Bus(const BusBase& other) : BusBase(other) {}
  explicit Bus(BusBase&& other) : BusBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static GType get_type_func() { return GST_TYPE_BUS; }
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__BUS_HPP */
