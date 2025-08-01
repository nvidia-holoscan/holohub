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

#ifndef SLANG_UTILS_HPP
#define SLANG_UTILS_HPP

#include <slang-com-helper.h>
#include <holoscan/logger/logger.hpp>

namespace holoscan::ops {

/**
 * @brief Converts a Slang result code to a human-readable error string
 *
 * @param result The Slang result code to convert
 * @return const char* A string describing the error, or "Unknown error" if the result is not
 * recognized
 */
const char* get_error_string(Slang::Result result);

/**
 * @brief Macro for safe Slang API calls with automatic error handling
 *
 * This macro executes a Slang statement and automatically checks for errors.
 * If the call fails, it throws a std::runtime_error with detailed information
 * including the statement, line number, file name, and error description.
 *
 * Usage:
 *   SLANG_CALL(slangCreateSession(&session));
 *
 * @param stmt The Slang statement to execute
 * @return Slang::Result The result of the statement execution
 * @throws std::runtime_error if the Slang call fails
 */
#define SLANG_CALL(stmt)                                                            \
  ({                                                                                \
    Slang::Result _slang_result = stmt;                                             \
    if (SLANG_FAILED(_slang_result)) {                                              \
      throw std::runtime_error(                                                     \
          fmt::format("Slang call {} in line {} of file {} failed with '{}' ({}).", \
                      #stmt,                                                        \
                      __LINE__,                                                     \
                      __FILE__,                                                     \
                      get_error_string(_slang_result),                              \
                      static_cast<int>(_slang_result)));                            \
    }                                                                               \
    _slang_result;                                                                  \
  })

/**
 * @brief Macro for safe Slang API calls with diagnostics support
 *
 * This macro is similar to SLANG_CALL but includes diagnostic information
 * in the error message when available. It's useful for compilation errors
 * where additional diagnostic details are provided by Slang.
 *
 * Usage:
 *   SLANG_CALL_WITH_DIAGNOSTICS(slangCompileRequestCompile(request, diagnostics_blob));
 *
 * @param stmt The Slang statement to execute
 * @return Slang::Result The result of the statement execution
 * @throws std::runtime_error if the Slang call fails, including diagnostic information
 */
#define SLANG_CALL_WITH_DIAGNOSTICS(stmt)                                               \
  ({                                                                                    \
    Slang::Result _slang_result = stmt;                                                 \
    if (SLANG_FAILED(_slang_result)) {                                                  \
      throw std::runtime_error(fmt::format(                                             \
          "Slang call {} in line {} of file {} failed with '{}' ({}), diagnostics: {}", \
          #stmt,                                                                        \
          __LINE__,                                                                     \
          __FILE__,                                                                     \
          get_error_string(_slang_result),                                              \
          static_cast<int>(_slang_result),                                              \
          (const char*)diagnostics_blob->getBufferPointer()));                          \
    }                                                                                   \
    _slang_result;                                                                      \
  })

/**
 * @brief Macro to log Slang compilation diagnostics if available
 *
 * This macro checks if diagnostic information is available and logs it
 * at INFO level. It's typically used after compilation operations to
 * provide additional debugging information.
 *
 * Usage:
 *   SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);
 *
 * @param diagnostics_blob Pointer to the diagnostics blob containing compilation information
 */
#define SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob)                        \
  if (diagnostics_blob) {                                                 \
    HOLOSCAN_LOG_INFO("Slang compilation diagnostics: {}",                \
                      (const char*)diagnostics_blob->getBufferPointer()); \
  }

/**
 * Splits a string into a pair of strings, separated by a separator.
 *
 * @param s The string to split
 * @param separator The separator to use
 * @return A pair of strings, the first is the part before the colon, the second is the part after
 */
std::pair<std::string, std::string> split(const std::string& s, char separator);

/**
 * @brief Converts a string to lowercase
 *
 * @param s The string to convert
 * @return The lowercase string
 */
std::string to_lower(const std::string& s);

/**
 * @brief Converts a string to a specific type
 *
 * @tparam typeT The type to convert to
 * @param s The string to convert
 * @return The converted value
 */
template <typename typeT>
typeT from_string(const std::string& s);

/**
 * @brief Converts a string to a boolean
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
bool from_string(const std::string& s);

/**
 * @brief Converts a string to an int8_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int8_t from_string(const std::string& s);

/**
 * @brief Converts a string to a uint8_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint8_t from_string(const std::string& s);

/**
 * @brief Converts a string to an int16_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int16_t from_string(const std::string& s);

/**
 * @brief Converts a string to a uint16_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint16_t from_string(const std::string& s);

/**
 * @brief Converts a string to an int32_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int32_t from_string(const std::string& s);

/**
 * @brief Converts a string to a uint32_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint32_t from_string(const std::string& s);

/**
 * @brief Converts a string to an int64_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
int64_t from_string(const std::string& s);

/**
 * @brief Converts a string to a uint64_t
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
uint64_t from_string(const std::string& s);

/**
 * @brief Converts a string to a float
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
float from_string(const std::string& s);

/**
 * @brief Converts a string to a double
 *
 * @param s The string to convert
 * @return The converted value
 */
template <>
double from_string(const std::string& s);

}  // namespace holoscan::ops

#endif /* SLANG_UTILS_HPP */
