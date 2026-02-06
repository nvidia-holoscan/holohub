/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef APPLICATIONS_GSTREAMER_COMMON_ARG_PARSER_HPP
#define APPLICATIONS_GSTREAMER_COMMON_ARG_PARSER_HPP

#include <string>
#include <stdexcept>
#include <cstdint>

namespace holoscan {
namespace gstreamer {
namespace common {

/**
 * @brief Convert string to numeric type (template specializations)
 *
 * @tparam T Numeric type to convert to
 * @param str String to convert
 * @param pos Pointer to size_t to store position after conversion
 * @return Converted value
 * @throws std::invalid_argument or std::out_of_range on conversion failure
 */
template<typename T>
T string_to(const std::string& str, size_t* pos);

// Specialization for int
template<>
inline int string_to<int>(const std::string& str, size_t* pos) {
  return std::stoi(str, pos);
}

// Specialization for int64_t
template<>
inline int64_t string_to<int64_t>(const std::string& str, size_t* pos) {
  return std::stoll(str, pos);
}

/**
 * @brief Safely parse a numeric value with validation
 *
 * Validates both the format (entire string must be consumed) and range of the parsed value.
 * Provides detailed error messages that include the parameter name and problematic value.
 *
 * @tparam T Numeric type to parse (int, int64_t, etc.)
 * @param value_str String to parse
 * @param param_name Parameter name for error messages
 * @param min_value Minimum allowed value (inclusive)
 * @param max_value Maximum allowed value (inclusive)
 * @return Parsed and validated value
 * @throws std::invalid_argument if parsing fails or value is out of range
 *
 * @example
 * ```cpp
 * int width = parse_validated<int>("1920", "width", 64, 8192);  // Returns 1920
 * int width = parse_validated<int>("10", "width", 64, 8192);    // Throws: out of range
 * int width = parse_validated<int>("abc", "width", 64, 8192);   // Throws: invalid format
 * ```
 */
template<typename T>
T parse_validated(const std::string& value_str, const std::string& param_name,
                  T min_value, T max_value) {
  try {
    size_t pos = 0;
    T value = string_to<T>(value_str, &pos);

    // Check if entire string was consumed
    if (pos != value_str.length()) {
      throw std::invalid_argument("Invalid characters in value");
    }

    // Validate range
    if (value < min_value || value > max_value) {
      throw std::invalid_argument(
          "Value " + std::to_string(value) + " is out of range [" +
          std::to_string(min_value) + ", " + std::to_string(max_value) + "]");
    }

    return value;
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(
        "Invalid " + param_name + ": " + value_str + " (" + e.what() + ")");
  } catch (const std::out_of_range& e) {
    throw std::invalid_argument(
        param_name + " value is too large: " + value_str);
  }
}

}  // namespace common
}  // namespace gstreamer
}  // namespace holoscan

#endif  // APPLICATIONS_GSTREAMER_COMMON_ARG_PARSER_HPP
