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

#include "iterator.hpp"
#include <stdexcept>

namespace holoscan {
namespace gst {

// ============================================================================
// Iterator Implementation
// ============================================================================

Iterator::Iterator(::GstIterator* iterator) 
    : iterator_(iterator), current_item_(G_VALUE_INIT), last_result_(GST_ITERATOR_OK) {
  // Immediately advance to first item if iterator is valid
  if (iterator_) {
    last_result_ = gst_iterator_next(iterator_, &current_item_);
  } else {
    last_result_ = GST_ITERATOR_ERROR;
  }
}

Iterator::~Iterator() {
  // Always unset the GValue since we initialize it in constructor
  g_value_unset(&current_item_);
  if (iterator_) {
    gst_iterator_free(iterator_);
  }
}

Iterator::Iterator(Iterator&& other) noexcept 
    : iterator_(other.iterator_), 
      current_item_(other.current_item_),
      last_result_(other.last_result_) {
  other.iterator_ = nullptr;
  other.current_item_ = G_VALUE_INIT;
}

Iterator& Iterator::operator=(Iterator&& other) noexcept {
  if (this != &other) {
    // Clean up current state
    g_value_unset(&current_item_);
    if (iterator_) {
      gst_iterator_free(iterator_);
    }
    
    // Move from other
    iterator_ = other.iterator_;
    current_item_ = other.current_item_;
    last_result_ = other.last_result_;
    
    // Reset other
    other.iterator_ = nullptr;
    other.current_item_ = G_VALUE_INIT;
  }
  return *this;
}

void Iterator::operator++() {
  if (!iterator_) {
    throw std::runtime_error("Cannot advance invalid iterator");
  }
  
  // Reset previous value and advance to next
  g_value_reset(&current_item_);
  last_result_ = gst_iterator_next(iterator_, &current_item_);
}

const ::GValue& Iterator::operator*() const {
  return current_item_;
}

Iterator::operator bool() const {
  return iterator_ != nullptr && last_result_ == GST_ITERATOR_OK;
}

}  // namespace gst
}  // namespace holoscan

