/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PVA_UNSHARP_MASK_HPP
#define PVA_UNSHARP_MASK_HPP

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

class PvaUnsharpMaskImpl;

class PvaUnsharpMask {
 public:
  PvaUnsharpMask();
  ~PvaUnsharpMask();
  int32_t init(int32_t width, int32_t height, int32_t inputLinePitch, int32_t outputLinePitch);
  int32_t process(uint8_t* src, uint8_t* dst);
  bool isInitialized() const;

 private:
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_inputLinePitch;
  uint32_t m_outputLinePitch;
  bool m_initialized;
  std::unique_ptr<PvaUnsharpMaskImpl> pImpl;
};

#endif  // PVA_UNSHARP_MASK_HPP
