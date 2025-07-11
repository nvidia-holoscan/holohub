/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.cuh>

namespace holoscan::ops::orsi {
namespace segmentation_postprocessor {

using holoscan::ops::segmentation_postprocessor::Shape;


// TODO: to verify if this function is actually needed. Ideally replace with NPP implementation.
void cuda_resize(Shape input_shape, Shape output_shape, const uint8_t* input, uint8_t* output,
                 int32_t offset_x, int32_t offset_y);

}  // namespace segmentation_postprocessor
}  // namespace holoscan::ops::orsi
