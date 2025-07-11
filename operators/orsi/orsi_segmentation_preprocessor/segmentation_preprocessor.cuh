/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <driver_types.h>
#include <cstdint>

namespace holoscan::ops::orsi {
namespace segmentation_preprocessor {

struct Shape {
  int32_t height;
  int32_t width;
  int32_t channels;
};

enum DataFormat {
  kNCHW,
  kHWC,
  kNHWC,
};

typedef uint8_t output_type_t;

static constexpr size_t kMaxChannelCount = std::numeric_limits<output_type_t>::max();

void cuda_preprocess(enum DataFormat data_format, Shape shape, const float* input, float* output,
                                                                       float* means, float* stds);


}  // namespace segmentation_preprocessor
}  // namespace holoscan::ops::orsi
