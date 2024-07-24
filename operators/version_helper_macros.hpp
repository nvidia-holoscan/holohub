/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef VERSION_HELPER_MACROS_HPP
#define VERSION_HELPER_MACROS_HPP

// If GXF has gxf/std/dlpack_utils.hpp it has DLPack support
#if __has_include("gxf/std/dlpack_utils.hpp")
  #define GXF_HAS_DLPACK_SUPPORT 1
  #include "gxf/std/tensor.hpp"
#else
  #define GXF_HAS_DLPACK_SUPPORT 0
  #include "holoscan/core/gxf/gxf_tensor.hpp"
#endif

#endif /* VERSION_HELPER_MACROS_HPP */
