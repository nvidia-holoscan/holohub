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

#ifndef HOLOSCAN__GSTREAMER__GST__ELEMENT_HPP
#define HOLOSCAN__GSTREAMER__GST__ELEMENT_HPP

#include <gst/gst.h>
#include <string>
#include <type_traits>

#include "bus.hpp"
#include "object.hpp"
#include "pad.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Templated base class for GStreamer element types (GstElement and its derivatives)
 * @tparam Derived The derived class type (for CRTP pattern)
 * @tparam NativeType The GStreamer element type (e.g. ::GstElement, specialized elements)
 *
 * This template provides common element functionality for all GStreamer elements.
 * Derive concrete classes like Element from this base class.
 */
template <typename Derived, typename NativeType>
class ElementBase : public ObjectBase<Derived, NativeType> {
 public:
  explicit ElementBase(NativeType* element = nullptr) : ObjectBase<Derived, NativeType>(element) {}

  // Implicit conversion from base ObjectBase type
  explicit ElementBase(const ObjectBase<Derived, NativeType>& other)
      : ObjectBase<Derived, NativeType>(other) {}
  explicit ElementBase(ObjectBase<Derived, NativeType>&& other)
      : ObjectBase<Derived, NativeType>(std::move(other)) {}

  /// @brief Get the bus from this element
  /// @returns A Bus wrapper for the element's bus
  /// @note Returns owned reference from gst_element_get_bus (no sinking needed)
  Bus get_bus() const { return Bus(gst_element_get_bus(GST_ELEMENT(this->get()))); }

  /// @brief Set the state of this element
  /// @param state The target GstState (e.g., GST_STATE_NULL, GST_STATE_PLAYING)
  /// @returns The result of the state change operation
  GstStateChangeReturn set_state(GstState state) {
    return gst_element_set_state(GST_ELEMENT(this->get()), state);
  }

  /// @brief Get the state of this element
  /// @param state Pointer to store the current state (can be nullptr)
  /// @param pending Pointer to store the pending state (can be nullptr)
  /// @param timeout Timeout in nanoseconds (GST_CLOCK_TIME_NONE for infinite, 0 for immediate)
  /// @returns The result of the state query operation
  GstStateChangeReturn get_state(GstState* state, GstState* pending,
                                  GstClockTime timeout = GST_CLOCK_TIME_NONE) const {
    return gst_element_get_state(GST_ELEMENT(this->get()), state, pending, timeout);
  }

  /// @brief Link this element to another element
  /// @param dest The destination element to link to
  /// @returns true if linking was successful, false otherwise
  template <typename T>
  bool link(const T& dest) const {
    // Type errors from misuse will surface clearly from the if constexpr branches
    if constexpr (std::is_convertible_v<std::decay_t<T>, ::GstElement*>) {
      return gst_element_link(GST_ELEMENT(this->get()), static_cast<::GstElement*>(dest));
    } else {
      return gst_element_link(GST_ELEMENT(this->get()), GST_ELEMENT(dest.get()));
    }
  }

  /// @brief Link this element to multiple elements in sequence
  /// @param elements Variable number of elements to link in chain: this -> elem1 -> elem2 -> ...
  /// @returns true if all linking operations were successful, false otherwise
  template <typename... Elements>
  bool link_many(Elements&&... elements) const {
    if (!this->get()) {
      return false;
    }
    return gst_element_link_many(
        GST_ELEMENT(this->get()), get_gst_element(std::forward<Elements>(elements))..., nullptr);
  }

  /// @brief Get the name of this element
  /// @returns The name of the element as a string
  /// @note Properly handles memory allocation from gst_element_get_name()
  std::string get_name() const {
    gchar* name = gst_element_get_name(GST_ELEMENT(this->get()));
    if (!name) {
      return "";
    }
    std::string result(name);
    g_free(name);
    return result;
  }

  /// @brief Get a static pad from this element
  /// @param name The name of the pad (e.g., "src", "sink")
  /// @returns A Pad wrapper for the static pad, or empty Pad if not found
  /// @note Returns owned reference from gst_element_get_static_pad (no sinking needed)
  Pad get_static_pad(const std::string& name) const {
    return Pad(gst_element_get_static_pad(GST_ELEMENT(this->get()), name.c_str()));
  }

 private:
  // Helper function to extract GstElement* from ElementBase-derived classes or raw pointers
  template <typename T>
  static ::GstElement* get_gst_element(T&& element) {
    using DecayedT = std::decay_t<T>;

    // If it's a raw ::GstElement* pointer, use directly
    if constexpr (std::is_convertible_v<DecayedT, ::GstElement*>) {
      return static_cast<::GstElement*>(element);
    } else {
      // Otherwise, assume it's an ElementBase-derived class and call .get()
      return GST_ELEMENT(element.get());
    }
  }
};

/**
 * @brief Wrapper class for ::GstElement
 */
class Element : public ElementBase<Element, ::GstElement> {
 public:
  explicit Element(::GstElement* element = nullptr) : ElementBase(element) {}

  // Implicit conversion from base ElementBase type
  explicit Element(const ElementBase& other) : ElementBase(other) {}
  explicit Element(ElementBase&& other) : ElementBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static GType get_type_func() { return GST_TYPE_ELEMENT; }
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__ELEMENT_HPP */
