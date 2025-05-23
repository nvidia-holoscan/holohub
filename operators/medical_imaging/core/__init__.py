#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Core functionality for medical imaging operators.

This module provides the fundamental building blocks for medical imaging operations,
including application context management, data types, and model interfaces.

.. autosummary::
    :toctree: _autosummary

    AppContext
    Application
    DataPath
    Image
    IOType
    ModelFactory
    Model
    NamedModel
    TorchScriptModel
    TritonModel
    RuntimeEnv
    parse_args
"""

from holoscan.core import Application

from .app_context import AppContext, init_app_context
from .arg_parser import parse_args
from .domain.datapath import DataPath
from .domain.image import Image
from .io_type import IOType
from .models.factory import ModelFactory
from .models.model import Model
from .models.named_model import NamedModel
from .models.torch_model import TorchScriptModel
from .models.triton_model import TritonModel
from .runtime_env import RuntimeEnv

# Add the function to the existing Application class, which could've been used as helper func too.
# It is well understood that deriving from the Application base is a better approach, but maybe later.
Application.init_app_context = init_app_context


__all__ = [
    "AppContext",
    "Application",
    "DataPath",
    "Image",
    "IOType",
    "ModelFactory",
    "Model",
    "NamedModel",
    "TorchScriptModel",
    "TritonModel",
    "RuntimeEnv",
    "parse_args",
]
