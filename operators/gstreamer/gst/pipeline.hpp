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

#ifndef HOLOSCAN__GSTREAMER__GST__PIPELINE_HPP
#define HOLOSCAN__GSTREAMER__GST__PIPELINE_HPP

#include <gst/gst.h>

#include <type_traits>

#include "element.hpp"
#include "object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Templated base class for GStreamer bin types (GstBin, GstPipeline, etc.)
 * @tparam Derived The derived class type (for CRTP pattern)
 * @tparam NativeType The GStreamer bin type (e.g. ::GstBin, ::GstPipeline)
 *
 * This template provides common bin functionality for containers that can hold elements.
 * Derive concrete classes like Pipeline from this base class.
 */
template <typename Derived, typename NativeType>
class BinBase : public ElementBase<Derived, NativeType> {
 public:
  explicit BinBase(NativeType* bin = nullptr) : ElementBase<Derived, NativeType>(bin) {}

  // Implicit conversion from base ElementBase type
  explicit BinBase(const ElementBase<Derived, NativeType>& other)
      : ElementBase<Derived, NativeType>(other) {}
  explicit BinBase(ElementBase<Derived, NativeType>&& other)
      : ElementBase<Derived, NativeType>(std::move(other)) {}

  /**
   * @brief Add a single element to the bin
   * @tparam Element Template for Element type
   * @param element Element object to add
   * @return true if the element was successfully added, false otherwise
   *
   * This is a type-safe C++ wrapper around gst_bin_add().
   * Accepts an Element wrapper object or raw GstElement* pointer.
   *
   * Example: bin.add(element);
   */
  template <typename Element>
  bool add(const Element& element) {
    return gst_bin_add(GST_BIN(this->get()), get_gst_element(element));
  }

  /**
   * @brief Add multiple elements to the bin
   * @tparam Elements Variadic template for Element types
   * @param elements Variable number of Element objects to add
   *
   * This is a type-safe C++ wrapper around gst_bin_add_many().
   * Accepts any number of Element wrapper objects and adds them to the bin.
   *
   * Example: bin.add_many(element1, element2, element3);
   */
  template <typename... Elements>
  void add_many(const Elements&... elements) {
    static_assert(sizeof...(elements) > 0, "add_many requires at least one element");

    // Extract raw GstElement* from each Element wrapper and call gst_bin_add_many
    // The function expects a NULL-terminated list
    gst_bin_add_many(GST_BIN(this->get()), get_gst_element(elements)..., nullptr);
  }

  /**
   * @brief Get an element from the bin by name
   * @param name The name of the element to retrieve
   * @return Element wrapper for the found element, or empty Element if not found
   * @note Returns owned reference from gst_bin_get_by_name (no sinking needed)
   *
   * Example: auto element = bin.get_by_name("my_element");
   */
  Element get_by_name(const std::string& name) const {
    return Element(gst_bin_get_by_name(GST_BIN(this->get()), name.c_str()));
  }

 private:
  // Helper function to extract GstElement* from ElementBase-derived classes or raw pointers
  template <typename T>
  static ::GstElement* get_gst_element(const T& element) {
    using DecayedT = std::decay_t<T>;

    // If it's a raw ::GstElement* pointer, use directly
    if constexpr (std::is_convertible_v<DecayedT, ::GstElement*>) {
      return static_cast<::GstElement*>(element);
    } else {
      // Otherwise, assume it's an ElementBase-derived class and call .get()
      return element.get();
    }
  }
};

/**
 * @brief Wrapper class for ::GstPipeline
 */
class Pipeline : public BinBase<Pipeline, ::GstPipeline> {
 public:
  explicit Pipeline(::GstPipeline* pipeline = nullptr) : BinBase(pipeline) {}

  /// @brief Create a new pipeline with the given name
  /// @param name Name for the pipeline (passed to gst_pipeline_new)
  /// @returns A new Pipeline object with the floating reference sunk
  /// @throws std::runtime_error if pipeline creation fails
  static Pipeline create(const std::string& name) {
    Pipeline pipeline(GST_PIPELINE(gst_pipeline_new(name.c_str())));
    if (!pipeline.get()) {
      throw std::runtime_error("Failed to create GStreamer pipeline: " + name);
    }
    return pipeline.ref_sink();
  }

  // Implicit conversion from base BinBase type
  explicit Pipeline(const BinBase& other) : BinBase(other) {}
  explicit Pipeline(BinBase&& other) : BinBase(std::move(other)) {}

  // Provide GType for type-safe casting
  static GType get_type_func() { return GST_TYPE_PIPELINE; }
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__PIPELINE_HPP */
