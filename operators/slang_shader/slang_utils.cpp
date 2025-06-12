/*
 * SPDXFileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDXLicenseIdentifier: Apache2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE2.0
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

}  // namespace holoscan::ops
