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

#include "slang_utils.hpp"

namespace holoscan::ops {

const char* get_error_string(Slang::Result result) {
  switch (result) {
    case SLANG_E_NOT_IMPLEMENTED:
      return "SLANG_E_NOT_IMPLEMENTED";
    case SLANG_E_NO_INTERFACE:
      return "SLANG_E_NO_INTERFACE";
    case SLANG_E_ABORT:
      return "SLANG_E_ABORT";
    case SLANG_E_INVALID_HANDLE:
      return "SLANG_E_INVALID_HANDLE";
    case SLANG_E_INVALID_ARG:
      return "SLANG_E_INVALID_ARG";
    case SLANG_E_OUT_OF_MEMORY:
      return "SLANG_E_OUT_OF_MEMORY";
    case SLANG_E_BUFFER_TOO_SMALL:
      return "SLANG_E_BUFFER_TOO_SMALL";
    case SLANG_E_UNINITIALIZED:
      return "SLANG_E_UNINITIALIZED";
    case SLANG_E_PENDING:
      return "SLANG_E_PENDING";
    case SLANG_E_CANNOT_OPEN:
      return "SLANG_E_CANNOT_OPEN";
    case SLANG_E_NOT_FOUND:
      return "SLANG_E_NOT_FOUND";
    case SLANG_E_INTERNAL_FAIL:
      return "SLANG_E_INTERNAL_FAIL";
    case SLANG_E_NOT_AVAILABLE:
      return "SLANG_E_NOT_AVAILABLE";
    case SLANG_E_TIME_OUT:
      return "SLANG_E_TIME_OUT";
    default:
      return "Unknown Slang error";
  }
}

std::pair<std::string, std::string> split(const std::string& s, char separator) {
  auto colon_pos = s.find(separator);
  if (colon_pos != std::string::npos) {
    return {s.substr(0, colon_pos), s.substr(colon_pos + 1)};
  } else {
    return {s, ""};
  }
}

std::string to_lower(const std::string& s) {
  std::string s_lower = s;
  std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return s_lower;
}

/**
 * @brief Converts a string to a boolean
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
bool from_string(const std::string& s) {
  const std::string s_lower = to_lower(s);
  if (s_lower == "true") {
    return true;
  } else if (s_lower == "false") {
    return false;
  } else {
    throw std::runtime_error(fmt::format("Invalid boolean value: {}", s));
  }
}

/**
 * @brief Converts a string to an int8_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int8_t from_string(const std::string& s) {
  return static_cast<int8_t>(std::stoi(s));
}

/**
 * @brief Converts a string to a uint8_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint8_t from_string(const std::string& s) {
  return static_cast<uint8_t>(std::stoul(s));
}

/**
 * @brief Converts a string to an int16_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int16_t from_string(const std::string& s) {
  return static_cast<int16_t>(std::stoi(s));
}

/**
 * @brief Converts a string to a uint16_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint16_t from_string(const std::string& s) {
  return static_cast<uint16_t>(std::stoul(s));
}

/**
 * @brief Converts a string to an int32_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int32_t from_string(const std::string& s) {
  return static_cast<int32_t>(std::stoi(s));
}

/**
 * @brief Converts a string to a uint32_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint32_t from_string(const std::string& s) {
  return static_cast<uint32_t>(std::stoul(s));
}

/**
 * @brief Converts a string to an int64_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int64_t from_string(const std::string& s) {
  return static_cast<int64_t>(std::stoll(s));
}

/**
 * @brief Converts a string to a uint64_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint64_t from_string(const std::string& s) {
  return static_cast<uint64_t>(std::stoull(s));
}

/**
 * @brief Converts a string to a float
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
float from_string(const std::string& s) {
  return static_cast<float>(std::stof(s));
}

/**
 * @brief Converts a string to a double
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
double from_string(const std::string& s) {
  return static_cast<double>(std::stod(s));
}

}  // namespace holoscan::ops
