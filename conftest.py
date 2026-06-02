# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest


@pytest.fixture
def app():
    return pytest.importorskip("holoscan.core").Application()


@pytest.fixture
def fragment():
    return pytest.importorskip("holoscan.core").Fragment()


@pytest.fixture
def config_file():
    default_filename = "operator_parameters.yaml"
    default_directory = os.path.dirname(__file__)

    def _factory(filename=default_filename, directory=default_directory):
        return os.path.join(directory, filename)

    return _factory


@pytest.fixture
def mock_image():
    def _factory(shape, dtype="uint8", backend="cupy", seed=None):
        if backend == "cupy":
            try:
                # CuPy can be installed yet unimportable (no GPU/driver, ABI
                # mismatch), so skip on any import failure, not just ImportError.
                import cupy as xp
            except Exception as exc:
                pytest.skip(f"CuPy is unavailable in this test environment: {exc}")
        elif backend == "numpy":
            xp = np
        else:
            raise ValueError(f"Unknown backend: {backend}")
        rng = xp.random.default_rng(seed)
        dtype = xp.dtype(dtype)
        if dtype.kind in "ui":
            img = rng.integers(0, 256, size=shape, dtype=dtype, endpoint=False)
        elif dtype.kind == "f":
            img = rng.uniform(0.0, 1.0, size=shape)
            if img.dtype != dtype:
                img = img.astype(dtype, copy=False)
        else:
            raise ValueError(f"{dtype=} unsupported")
        return img

    return _factory


class MockOpInput:
    def __init__(self, tensor, tensor_name="", port=""):
        self._tensor = tensor
        self._tensor_name = tensor_name
        self._port = port

    def receive(self, port):
        assert port == self._port
        return {self._tensor_name: self._tensor}


class MockOpOutput:
    def __init__(self):
        self.emitted = None

    def emit(self, msg, port):
        self.emitted = (msg, port)


@pytest.fixture
def op_input_factory():
    def _factory(tensor, tensor_name="", port=""):
        return MockOpInput(tensor, tensor_name=tensor_name, port=port)

    return _factory


@pytest.fixture
def op_output():
    return MockOpOutput()


@pytest.fixture
def execution_context():
    return None
