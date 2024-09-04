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
.. autosummary::
    :toctree: _autosummary

    AppContext
    IOType
    InputContext
    OutputContext
    RuntimeEnv
    init_app_context
    parse_args
"""

# Need to import explicit ones to quiet mypy complaints
from holoscan.core import Application

from .app_context import AppContext as AppContext
from .app_context import init_app_context
from .arg_parser import parse_args as parse_args
from .domain.datapath import DataPath as DataPath
from .domain.image import Image as Image
from .io_type import IOType as IOType
from .models.factory import ModelFactory as ModelFactory
from .models.model import Model as Model
from .models.named_model import NamedModel as NamedModel
from .models.torch_model import TorchScriptModel as TorchScriptModel
from .models.triton_model import TritonModel as TritonModel
from .runtime_env import RuntimeEnv as RuntimeEnv

# Add the function to the existing Application class, which could've been used as helper func too.
# It is well understood that deriving from the Application base is a better approach, but maybe later.
Application.init_app_context = init_app_context
