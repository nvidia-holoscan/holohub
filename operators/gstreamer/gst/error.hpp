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

#ifndef HOLOSCAN__GSTREAMER__GST__ERROR_HPP
#define HOLOSCAN__GSTREAMER__GST__ERROR_HPP

#include <memory>
#include <glib.h>

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GError with automatic cleanup
 * 
 * This class manages the lifetime of a GError object and automatically
 * calls g_error_free when destroyed.
 * 
 * Note: GError uses copy/free semantics, not ref-counting, so it doesn't
 * inherit from Object<T>.
 */
class Error {
 public:
  /**
   * @brief Constructor from raw pointer (takes ownership)
   * @param error GError pointer to wrap (nullptr is allowed)
   */
  explicit Error(::GError* error = nullptr) : ptr_(error, [](::GError* err) {
    if (err) {
      g_error_free(err);
    }
  }) {}

  /**
   * @brief Get the raw pointer
   * @return Raw GError pointer
   */
  ::GError* get() const { return ptr_.get(); }

  /**
   * @brief Access GError members directly
   * @return Pointer to GError for member access
   */
   const ::GError* operator->() const { return ptr_.get(); }

  /**
   * @brief Bool conversion
   * @return true if error is not nullptr
   */
  explicit operator bool() const { return static_cast<bool>(ptr_); }

 private:
  std::shared_ptr<::GError> ptr_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__ERROR_HPP */

