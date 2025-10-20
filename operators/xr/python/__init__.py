"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import holoscan.core  # noqa: F401

from ._xr import (
    ViewConfigurationDepthRangeEXT,
    XrBeginFrameOp,
    XrCompositionLayerProjectionStorage,
    XrCompositionLayerProjectionView,
    XrEndFrameOp,
    XrFovf,
    XrFrameState,
    XrPosef,
    XrQuaternionf,
    XrSession,
    XrSwapchainCuda,
    XrSwapchainCudaFormat,
    XrVector3f,
    XrViewConfigurationView,
)

__all__ = [
    "XrSession",
    "XrBeginFrameOp",
    "XrEndFrameOp",
    "XrSwapchainCuda",
    "XrSwapchainCudaFormat",
    "XrCompositionLayerProjectionStorage",
    "XrFrameState",
    "XrCompositionLayerProjectionView",
    "XrFovf",
    "XrPosef",
    "XrQuaternionf",
    "XrVector3f",
    "ViewConfigurationDepthRangeEXT",
    "XrViewConfigurationView",
]
