# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the DepthToPointCloudOp Python bindings.

Covers construction, port wiring, parameter handling, and error handling across
the pybind11 boundary. Numerical correctness of the deprojection kernel is
covered separately by the standalone CUDA golden-reference test
(``test/test_deproject.cu``) and the ``depth_to_point_cloud_demo`` application,
since the compute path requires real GXF entities, an allocator, and a CUDA
stream that the lightweight mock fixtures do not provide.
"""

import pytest
from holoscan.core import Operator
from holoscan.resources import UnboundedAllocator

from holohub.depth_to_point_cloud import DepthToPointCloudOp

try:
    from holoscan.core import BaseOperator
except ImportError:
    from holoscan.core import _Operator as BaseOperator


def _make_op(fragment, **overrides):
    """Construct a DepthToPointCloudOp with sane defaults, overridable per test."""
    params = dict(
        name="depth_to_point_cloud",
        allocator=UnboundedAllocator(fragment, name="alloc"),
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    params.update(overrides)
    return DepthToPointCloudOp(fragment, **params)


def test_init(fragment):
    """Operator constructs and exposes the expected Holoscan properties."""
    name = "d2p_op"
    op = _make_op(fragment, name=name)
    assert isinstance(op, BaseOperator), "DepthToPointCloudOp should be a Holoscan operator"
    assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
    assert f"name: {name}" in repr(op), "Operator name should appear in repr()"


def test_ports(fragment):
    """setup() wires the required depth input, optional inputs, and the output."""
    spec = _make_op(fragment).spec
    assert "depth" in spec.inputs, 'required input "depth" missing'
    assert "intrinsics" in spec.inputs, 'optional per-frame "intrinsics" input missing'
    assert "color" in spec.inputs, 'optional "color" input missing'
    assert "point_cloud" in spec.outputs, 'output "point_cloud" missing'


def test_requires_allocator(fragment):
    """allocator is a required argument; omitting it is a TypeError."""
    with pytest.raises(TypeError):
        DepthToPointCloudOp(fragment, name="no_alloc", fx=500.0, fy=500.0, cx=320.0, cy=240.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"depth_scale": 0.001, "depth_min": 0.1, "depth_max": 10.0},
        {"output_tensor_name": "xyz", "output_color_tensor_name": "rgb"},
        {"invalid_value": 0.0},
        {"depth_tensor_name": "depth", "color_tensor_name": "color"},
    ],
)
def test_param_acceptance(fragment, overrides):
    """The operator accepts its documented parameters without error."""
    op = _make_op(fragment, **overrides)
    assert isinstance(op, BaseOperator)
