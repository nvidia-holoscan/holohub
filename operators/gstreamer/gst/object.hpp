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

#ifndef HOLOSCAN__GSTREAMER__GST__OBJECT_HPP
#define HOLOSCAN__GSTREAMER__GST__OBJECT_HPP

#include <gst/gst.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "wrapper_base.hpp"

namespace holoscan {
namespace gst {

// Forward declaration
template <typename DerivedT, typename NativeTypeT>
class ObjectBase;

/**
 * @brief Type-safe cast between GStreamer object wrapper types
 * @tparam To Target wrapper type (e.g., Element, Bus, Pad)
 * @tparam From Source wrapper type (must be derived from ObjectBase)
 * @param from Source object wrapper to cast from
 * @return New wrapper instance of target type containing the safely casted pointer
 *
 * This function provides compile-time type-safe casting between GStreamer object wrappers,
 * similar to std::static_pointer_cast for shared_ptr. It uses GObject's type checking
 * system (g_type_check_instance_cast) to verify the cast is valid at runtime.
 *
 * @example
 *   Element elem = ...;
 *   auto bin = static_object_cast<Bin>(elem);  // Cast Element to Bin
 *
 * @note The To type must define get_type_func() returning its GType constant.
 * @note This function is only for GstObject-based wrappers (ObjectBase hierarchy).
 *       Using it with GstMiniObject-based wrappers (e.g., Caps, Buffer) is undefined behavior.
 */
template <typename To, typename From>
To static_object_cast(const From& from) {
  using ToNativeType = typename To::NativeType;

  // Compile-time check: Ensure both types are ObjectBase-derived (not MiniObjectBase)
  static_assert(
      std::is_base_of_v<ObjectBase<typename From::Derived, typename From::NativeType>, From>,
      "static_object_cast can only be used with ObjectBase-derived wrappers");
  static_assert(
      std::is_base_of_v<ObjectBase<typename To::Derived, typename To::NativeType>, To>,
      "static_object_cast can only be used with ObjectBase-derived wrappers");

  if (!from.get()) {
    return To(nullptr);
  }

  // Use GObject type checking to safely cast
  // Note: g_type_check_instance_cast returns GTypeInstance*, so we use C-style cast
  auto* casted = (ToNativeType*)(g_type_check_instance_cast((GTypeInstance*)(from.get()),
                                                            To::get_type_func()));

  // Create new wrapper with the casted pointer
  // Note: We need to ref since we're creating a new wrapper for existing object
  if (casted) {
    gst_object_ref(casted);
  }

  return To(casted);
}

/**
 * @brief Internal RAII base class for GStreamer GstObject types with automatic cleanup
 * @tparam DerivedT The derived class type (for CRTP pattern)
 * @tparam NativeTypeT The GStreamer GstObject type (e.g. ::GstElement, ::GstBus, ::GstPad, etc.)
 *
 * This template provides automatic reference counting and cleanup for GstObject-derived types.
 * All GstObject types use the same reference counting functions:
 * - gst_object_ref: Increment reference count
 * - gst_object_unref: Decrement reference count
 * - gst_object_ref_sink: Sink floating references
 *
 * @note This class is for GstObject hierarchy only (Element, Bus, Pad, Pipeline, Allocator, etc.).
 *       For GstMiniObject types (Buffer, Memory, Caps, Message), use MiniObjectBase from
 * mini_object.hpp. This class is designed for internal use only and has no virtual functions.
 */
template <typename DerivedT, typename NativeTypeT>
class ObjectBase : public GstWrapperBase {
 public:
  // Local type aliases for cleaner code inside the class
  using Derived = DerivedT;
  using NativeType = NativeTypeT;

  // Constructor from raw pointer (takes ownership)
  explicit ObjectBase(NativeType* object = nullptr)
      : ptr_(object, [](NativeType* obj) {
          if (obj)
            gst_object_unref(obj);
        }) {
    // No automatic sinking - use .ref_sink() method for floating references
  }

  ~ObjectBase() = default;

  // Enable copy semantics
  ObjectBase(const ObjectBase& other) = default;
  ObjectBase& operator=(const ObjectBase& other) = default;

  // Enable move semantics
  ObjectBase(ObjectBase&& other) = default;
  ObjectBase& operator=(ObjectBase&& other) = default;

  // Get the raw pointer
  NativeType* get() const { return ptr_.get(); }

  // Access GStreamer object members directly
  NativeType* operator->() const { return ptr_.get(); }

  // Bool conversion
  explicit operator bool() const { return static_cast<bool>(ptr_); }

  /**
   * @brief Increment GStreamer reference count (advanced: for external ownership transfer)
   * @return Reference to this object for chaining
   *
   * @warning This is an advanced escape hatch for transferring ownership to external GStreamer code.
   *          The caller is responsible for ensuring a matching gst_object_unref is called by the
   *          external code. The shared_ptr deleter will only call unref once when this wrapper is
   *          destroyed, so using ref() without external unref will leak GObject references.
   *
   * @note For most use cases, prefer copying the wrapper (which automatically manages refcounts)
   *       or using get() to pass the raw pointer to functions that don't take ownership.
   */
  Derived& ref() {
    if (ptr_)
      gst_object_ref(ptr_.get());
    return static_cast<Derived&>(*this);
  }

  /**
   * @brief Const overload of ref() for const objects
   * @return Const reference to this object for chaining
   *
   * @warning See non-const ref() for important usage notes about external ownership transfer.
   */
  const Derived& ref() const {
    if (ptr_)
      gst_object_ref(ptr_.get());
    return static_cast<const Derived&>(*this);
  }

  /**
   * @brief Sink floating reference
   * @return Reference to this object for chaining
   *
   * Use this method for objects created with floating references (e.g., from factory functions).
   * Do NOT use for objects returned by getter functions that already have owned references.
   *
   * @note All GstObject-derived types support floating references and can be sunk.
   */
  Derived& ref_sink() {
    if (ptr_) {
      gst_object_ref_sink(ptr_.get());
    }
    return static_cast<Derived&>(*this);
  }

  // Const overload for const objects
  const Derived& ref_sink() const {
    if (ptr_) {
      gst_object_ref_sink(ptr_.get());
    }
    return static_cast<const Derived&>(*this);
  }

 private:
  /**
   * @brief Helper to convert arguments for g_object_set
   * @tparam T Argument type (automatically deduced)
   * @param arg Argument to potentially convert
   * @return Converted argument suitable for g_object_set:
   *         - const char* for std::string arguments
   *         - Raw pointer for GStreamer wrapper objects (e.g., gst::Caps, gst::Element)
   *         - Forwarded original argument for all other types
   *
   * @note std::string arguments must be lvalues to avoid dangling pointers (temporaries are rejected
   *       at compile time). Conversions are resolved via `if constexpr` with minimal overhead.
   */
  template <typename T>
  static constexpr auto convert_for_gobject(T&& arg) {
    using DecayT = std::decay_t<T>;

    if constexpr (std::is_same_v<DecayT, std::string>) {
      // Convert std::string to const char* (must be lvalue to avoid dangling pointer)
      static_assert(std::is_lvalue_reference_v<T&&>,
                    "std::string arguments to set_properties must be lvalues; use a named "
                    "variable or .c_str() for temporaries.");
      return arg.c_str();
    } else if constexpr (std::is_base_of_v<GstWrapperBase, DecayT>) {
      // Automatically unwrap GStreamer wrapper objects (e.g., gst::Caps, gst::Element)
      return arg.get();
    } else {
      // Forward all other types unchanged
      return std::forward<T>(arg);
    }
  }

 public:
  /**
   * @brief Set multiple properties on the GObject in a single call
   * @tparam Args Variadic template for property names and values
   * @param args Property name-value pairs (name1, value1, name2, value2, ...)
   *
   * @note This variadic template function mimics g_object_set behavior:
   *       - Accepts any number of property name-value pairs
   *       - Automatically converts std::string to const char* at compile time
   *       - Automatically unwraps GStreamer wrapper objects (e.g., gst::Caps)
   *       - Supports all types that g_object_set can handle
   *       - Property names can be const char* or std::string (must be lvalues)
   *       - Values can be any supported GObject property type
   *       - Minimal overhead - conversions resolved at compile time
   *
   * @example
   *   // Set multiple properties at once
   *   element.set_properties("bitrate", 1000000,
   *                          "name", "my-element",
   *                          "enabled", true,
   *                          "quality", 1.5f);
   *
   *   // Works with std::string property names too (must be lvalues)
   *   std::string prop = "bitrate";
   *   std::string name = "test";
   *   element.set_properties(prop, 2000000, "name", name.c_str());
   *
   *   // Automatically unwraps GStreamer wrapper objects
   *   gst::Caps caps("video/x-raw,format=RGBA");
   *   element.set_properties("caps", caps);  // Automatically calls caps.get()
   */
  template <typename... Args>
  void set_properties(Args&&... args) {
    static_assert(sizeof...(Args) % 2 == 0,
                  "set_properties expects an even number of arguments: (name, value) pairs.");
    if (!ptr_)
      return;

    // Call g_object_set with compile-time converted arguments
    g_object_set(G_OBJECT(ptr_.get()), convert_for_gobject(std::forward<Args>(args))..., nullptr);
  }

  /**
   * @brief Find property type and set it from string value with automatic conversion
   * @param name Property name
   * @param value Property value as string
   * @return true if property was found and set successfully, false otherwise
   *
   * @note This function uses GObject introspection to:
   *       - Find the property specification and its expected type
   *       - Automatically convert the string value to the correct type
   *       - Set the property with the converted value
   *       - Handle type conversion errors gracefully with logging
   *
   * @example
   *   element.find_and_set_property("bitrate", "1000000");    // auto-converts to int
   *   element.find_and_set_property("name", "my-element");    // stays as string
   *   element.find_and_set_property("enabled", "true");       // auto-converts to bool
   */
  bool find_and_set_property(const std::string& name, const std::string& value) {
    if (!ptr_)
      return false;

    // Find the property specification
    GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(ptr_.get()), name.c_str());

    if (!pspec) {
      return false;
    }

    // Set property based on its type
    GType ptype = G_PARAM_SPEC_VALUE_TYPE(pspec);

    try {
      switch (ptype) {
        case G_TYPE_STRING:
          set_properties(name, value);
          break;

        case G_TYPE_INT: {
          int int_val = std::stoi(value);
          set_properties(name, int_val);
          break;
        }

        case G_TYPE_UINT: {
          unsigned int uint_val = std::stoul(value);
          set_properties(name, uint_val);
          break;
        }

        case G_TYPE_INT64: {
          int64_t int64_val = std::stoll(value);
          set_properties(name, int64_val);
          break;
        }

        case G_TYPE_UINT64: {
          uint64_t uint64_val = std::stoull(value);
          set_properties(name, uint64_val);
          break;
        }

        case G_TYPE_BOOLEAN: {
          // Support common boolean representations, case-insensitive
          std::string lower_val = value;
          std::transform(lower_val.begin(), lower_val.end(), lower_val.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
          bool bool_val = (lower_val == "true" || lower_val == "1" ||
                           lower_val == "yes" || lower_val == "on");
          set_properties(name, bool_val);
          break;
        }

        case G_TYPE_FLOAT: {
          float float_val = std::stof(value);
          set_properties(name, float_val);
          break;
        }

        case G_TYPE_DOUBLE: {
          double double_val = std::stod(value);
          set_properties(name, double_val);
          break;
        }

        default:
          fprintf(stderr,
                  "Unsupported property type for '%s' (type: %s), skipping\n",
                  name.c_str(),
                  g_type_name(ptype));
          return false;
      }
    } catch (const std::exception& e) {
      // Note: Using fprintf to avoid dependency on holoscan logging in base class
      fprintf(stderr,
              "Failed to convert '%s' for property '%s': %s\n",
              value.c_str(),
              name.c_str(),
              e.what());
      return false;
    }

    return true;
  }

  // Reset the guard
  void reset() { ptr_.reset(); }

 private:
  std::shared_ptr<NativeType> ptr_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__OBJECT_HPP */
