# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import cupy as cp
import pytest

from holoscan.core import Application, Fragment, OperatorSpec
from holoscan.logger import LogLevel, set_log_level

# set log level to INFO during testing
set_log_level(LogLevel.DEBUG)


@pytest.fixture
def app():
    return Application()


@pytest.fixture
def fragment():
    return Fragment()


@pytest.fixture
def config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "operator_parameters.yaml")
    return config_file


@pytest.fixture
def dummy_image_factory():
    def _factory(shape, dtype=cp.uint8):
        return cp.random.randint(0, 255, size=shape, dtype=dtype)
    return _factory

class DummyInput:
    def __init__(self, tensor, tensor_name="", port=""):
        self._tensor = tensor
        self._tensor_name = tensor_name
        self._port = port

    def receive(self, port):
        assert port == self._port
        return {self._tensor_name: self._tensor}


class DummyOutput:
    def __init__(self):
        self.emitted = None

    def emit(self, msg, port):
        self.emitted = (msg, port)


class DummyContext:
    pass


@pytest.fixture
def op_input_factory():
    def _factory(tensor, tensor_name="", port=""):
        return DummyInput(tensor, tensor_name=tensor_name, port=port)

    return _factory


@pytest.fixture
def op_output():
    return DummyOutput()


@pytest.fixture
def context():
    return DummyContext()
