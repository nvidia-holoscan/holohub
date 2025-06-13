/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ux_cursor.hpp"

namespace holoscan::openxr {

UxCursorController::UxCursorController(UxCursor& cursor)
    : cursor_(cursor), transform_(cursor.transform) {}

void UxCursorController::cursorMove(Eigen::Affine3f pose) {
  cursor_.transform = pose;
}

void UxCursorController::cursorClick(Eigen::Affine3f pose) {}

void UxCursorController::cursorRelease() {}
}  // namespace holoscan::openxr
