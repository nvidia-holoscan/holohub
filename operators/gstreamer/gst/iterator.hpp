/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef GST_ITERATOR_HPP
#define GST_ITERATOR_HPP

#include <gst/gst.h>

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstIterator with automatic cleanup and C++ iterator interface
 *
 * This class provides safe access to GStreamer iterators by automatically handling
 * the cleanup lifecycle and providing a C++-style iterator interface.
 */
class Iterator {
public:
  /**
   * @brief Constructor that takes ownership of a GstIterator
   * @param iterator GstIterator to wrap (takes ownership, can be nullptr)
   */
  explicit Iterator(::GstIterator* iterator);

  /**
   * @brief Destructor automatically frees the iterator
   */
  ~Iterator();

  // Delete copy operations to prevent double free
  Iterator(const Iterator&) = delete;
  Iterator& operator=(const Iterator&) = delete;

  // Allow move operations
  Iterator(Iterator&& other) noexcept;
  Iterator& operator=(Iterator&& other) noexcept;

  /**
   * @brief Advance iterator to next item (prefix increment)
   */
  void operator++();

  /**
   * @brief Get current item from iterator
   * @return Reference to current GValue (only valid when iterator is valid)
   */
  const ::GValue& operator*() const;

  /**
   * @brief Boolean conversion operator - check if current item is valid
   * @return true if current item can be safely accessed
   */
  explicit operator bool() const;

private:
  ::GstIterator* iterator_;
  ::GValue current_item_;
  ::GstIteratorResult last_result_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* GST_ITERATOR_HPP */

