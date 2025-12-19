# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module provides a Python API to underlying C++ API Operators.

.. autosummary::

    holoscan.operators.BayerDemosaicOp
    holoscan.operators.FormatConverterOp
    holoscan.operators.GXFCodeletOp
    holoscan.operators.HolovizOp
    holoscan.operators.InferenceOp
    holoscan.operators.InferenceProcessorOp
    holoscan.operators.PingRxOp
    holoscan.operators.PingTensorRxOp
    holoscan.operators.PingTensorTxOp
    holoscan.operators.PingTxOp
    holoscan.operators.SegmentationPostprocessorOp
    holoscan.operators.DataTypeTxTestOp
    holoscan.operators.V4L2VideoCaptureOp
    holoscan.operators.VideoStreamRecorderOp
    holoscan.operators.VideoStreamReplayerOp
"""

# Define operator modules and classes for lazy loading
_OPERATOR_MODULES = {
    "bayer_demosaic": ["BayerDemosaicOp"],
    "format_converter": ["FormatConverterOp"],
    "gxf_codelet": ["GXFCodeletOp"],
    "holoviz": ["HolovizOp"],
    "inference": ["InferenceOp"],
    "inference_processor": ["InferenceProcessorOp"],
    "ping_rx": ["PingRxOp"],
    "ping_tensor_rx": ["PingTensorRxOp"],
    "ping_tensor_tx": ["PingTensorTxOp"],
    "ping_tx": ["PingTxOp"],
    "segmentation_postprocessor": ["SegmentationPostprocessorOp"],
    # Test operators are not intended to have a stable API for external use, so do not automatically
    # import them to the public API here.
    # "test_ops": ["DataTypeRxTestOp", "DataTypeTxTestOp"],
    "v4l2_video_capture": ["V4L2VideoCaptureOp"],
    "video_stream_recorder": ["VideoStreamRecorderOp"],
    "video_stream_replayer": ["VideoStreamReplayerOp"],
}
_OPERATORS = [item for sublist in _OPERATOR_MODULES.values() for item in sublist]
_LOADED_OPERATORS = {}
# expose both operator modules and operator classes for backward compatibility
__all__ = list(_OPERATOR_MODULES.keys()) + _OPERATORS


# Autocomplete
def __dir__():
    return __all__


# Lazily load modules and classes
def __getattr__(attr):
    # Get submodule, import if needed
    def getsubmodule(name):
        import importlib
        import sys

        module_name = f"{__name__}.{name}"
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return module

    # Return submodule
    if attr in _OPERATOR_MODULES:
        return getsubmodule(attr)

    # Return cached operator class
    if attr in _LOADED_OPERATORS:
        return _LOADED_OPERATORS[attr]

    # Get new operator class
    if attr in _OPERATORS:
        # Search for submodule that holds it
        for module_name, values in _OPERATOR_MODULES.items():
            if attr in values:
                operator = getattr(getsubmodule(module_name), attr)  # retrieve from submodule
                _LOADED_OPERATORS[attr] = operator  # cache
                return operator

    raise AttributeError(f"module {__name__} has no attribute {attr}")
