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

#ifndef HOLOSCAN__GSTREAMER__GST__OBJECT_HPP
#define HOLOSCAN__GSTREAMER__GST__OBJECT_HPP

#include <gst/gst.h>

#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

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
 */
template <typename To, typename From>
To static_object_cast(const From& from) {
  using ToNativeType = typename To::NativeType;

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
class ObjectBase {
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

  // Increment GStreamer reference count and return reference for chaining
  Derived& ref() {
    if (ptr_)
      gst_object_ref(ptr_.get());
    return static_cast<Derived&>(*this);
  }

  // Const overload for const objects
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
   * @brief Compile-time helper to convert std::string to const char*, leave other types unchanged
   * @tparam T Argument type (automatically deduced)
   * @param arg Argument to potentially convert
   * @return const char* for std::string arguments, forwarded original argument for all other types
   *
   * @note This function is evaluated completely at compile time, generating zero runtime overhead
   */
  template <typename T>
  static constexpr auto convert_for_gobject(T&& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
      return arg.c_str();
    } else {
      return std::forward<T>(arg);
    }
  }

 public:
  /**
   * @brief Set multiple properties on the GObject in a single call
   * @tparam Args Variadic template for property names and values
   * @param args Property name-value pairs (name1, value1, name2, value2, ...)
   * @return true if properties were set successfully, false otherwise
   *
   * @note This variadic template function mimics g_object_set behavior:
   *       - Accepts any number of property name-value pairs
   *       - Automatically converts std::string to const char* at COMPILE TIME
   *       - Supports all types that g_object_set can handle
   *       - Property names can be const char* or std::string
   *       - Values can be any supported GObject property type
   *       - ZERO runtime overhead - all conversions resolved at compile time
   *
   * @example
   *   // Set multiple properties at once
   *   element.set_properties("bitrate", 1000000,
   *                          "name", "my-element",
   *                          "enabled", true,
   *                          "quality", 1.5f);
   *
   *   // Works with std::string property names too
   *   std::string prop = "bitrate";
   *   element.set_properties(prop, 2000000, "name", std::string("test"));
   */
  template <typename... Args>
  void set_properties(Args&&... args) {
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
          bool bool_val = (value == "true" || value == "1" || value == "TRUE" || value == "True");
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
