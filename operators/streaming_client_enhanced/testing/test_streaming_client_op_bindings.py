#!/usr/bin/env python3
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

"""
Unit tests for StreamingClientOp Python bindings (pybind11).

This module tests the Python bindings of the StreamingClientOp operator.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestStreamingClientOpBinding:
    """Test class for StreamingClientOp Python binding functionality."""

    @pytest.mark.unit
    def test_operator_creation_basic(self, operator_factory):
        """Test basic operator creation through Python bindings."""
        op = operator_factory()
        assert op is not None
        assert hasattr(op, 'initialize')
        assert hasattr(op, 'setup')

    @pytest.mark.unit
    def test_operator_creation_with_custom_name(self, operator_factory):
        """Test operator creation with custom name."""
        custom_name = "my_streaming_client"
        op = operator_factory(name=custom_name)
        assert op is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("width,height,fps", [
        (640, 480, 30),
        (1280, 720, 60),
        (1920, 1080, 30),
    ])
    def test_video_parameters(self, operator_factory, width, height, fps):
        """Test operator creation with different video parameters."""
        op = operator_factory(
            width=width,
            height=height,
            fps=fps
        )
        assert op is not None

    @pytest.mark.unit
    def test_parameter_type_validation(self, operator_factory):
        """Test that parameter types are properly validated."""
        op = operator_factory(
            width=640,
            height=480,
            fps=30,
            server_ip="127.0.0.1",
            signaling_port=48010,
            receive_frames=True,
            send_frames=False,
            min_non_zero_bytes=100
        )
        assert op is not None

    @pytest.mark.unit
    def test_method_availability(self, operator_factory):
        """Test that required methods are available through Python bindings."""
        op = operator_factory()
        
        # Check core operator methods
        assert hasattr(op, 'initialize')
        assert callable(getattr(op, 'initialize'))
        
        assert hasattr(op, 'setup')
        assert callable(getattr(op, 'setup'))

    @pytest.mark.unit
    def test_docstring_availability(self, streaming_client_op_class):
        """Test that docstrings are available for the Python bindings."""
        assert hasattr(streaming_client_op_class, '__doc__')
        doc = getattr(streaming_client_op_class, '__doc__')
        assert doc is not None
        assert len(doc.strip()) > 0
