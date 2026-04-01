/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef HOLOHUB_IMX274_GPU_RESIDENT_SHARED_FRAME_STATE
#define HOLOHUB_IMX274_GPU_RESIDENT_SHARED_FRAME_STATE

#include <cstddef>
#include <functional>
#include <mutex>

#include <cuda.h>

namespace imx274_gpu_resident {

struct SharedFrameState {
  mutable std::mutex mutex;
  std::function<CUdeviceptr()> get_frame_memory_base;
  size_t frame_size_rounded{0};
  unsigned char** chosen_frame_memory{nullptr};
};

}  // namespace imx274_gpu_resident

#endif /* HOLOHUB_IMX274_GPU_RESIDENT_SHARED_FRAME_STATE */
