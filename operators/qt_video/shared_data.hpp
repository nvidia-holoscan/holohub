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

#ifndef OPERATORS_QT_VIDEO_SHARED_DATA
#define OPERATORS_QT_VIDEO_SHARED_DATA

#include "qt_video_op.hpp"

#include <condition_variable>
#include <mutex>

// forward declarations
typedef struct CUevent_st* cudaEvent_t;
namespace nvidia::gxf {
struct VideoBufferInfo;
}

typedef struct QtHoloscanSharedData_t {
  std::mutex mutex_;
  std::condition_variable condition_variable_;

  void *pointer_ = nullptr;
  nvidia::gxf::VideoBufferInfo video_buffer_info_{};
  cudaEvent_t cuda_event_ = nullptr;

  enum class State { Unknown, Ready, Processed };
  State state_ = State::Unknown;
} QtHoloscanSharedData;

#endif /* OPERATORS_QT_VIDEO_SHARED_DATA */
