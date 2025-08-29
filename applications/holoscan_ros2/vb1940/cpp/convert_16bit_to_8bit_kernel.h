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

#ifndef CONVERT_16BIT_TO_8BIT_KERNEL_H
#define CONVERT_16BIT_TO_8BIT_KERNEL_H

#include <cstdint>

// C++ wrapper function declaration
extern "C" void launch_convert_16bit_to_8bit_kernel(const uint16_t* input, uint8_t* output,
                                                    int width, int height, int input_channels);

#endif  // CONVERT_16BIT_TO_8BIT_KERNEL_H
