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

import pytest
from holoscan.core import Operator

from .pixelator import PixelatorOp

try:
    from holoscan.code import BaseOperator
except ImportError:
    from holoscan.code import _Operator as BaseOperator


def test_pixelator_op_init(fragment):
    """Test PixelatorOp initialization and its properties."""
    name = "pixelator_op"
    op = PixelatorOp(fragment=fragment, name=name, tensor_name="image")
    assert isinstance(op, BaseOperator), "PixelatorOp should be a Holoscan operator"
    assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
    assert f"name: {name}" in repr(op), "Operator name should appear in repr()"


def test_pixelator_op_setup(fragment):
    """Test PixelatorOp setup for input/output ports."""
    op = PixelatorOp(fragment=fragment, tensor_name="image")
    spec = op.spec
    assert "in" in spec.inputs, 'Input port "in" should be present in spec.inputs'
    assert "out" in spec.outputs, 'Output port "out" should be present in spec.outputs'


def test_pixelator_op_invalid_block_size(fragment):
    """Test PixelatorOp raises error on invalid block sizes."""
    import pytest

    with pytest.raises(ValueError, match="block_size_h must be a positive integer"):
        PixelatorOp(fragment=fragment, tensor_name="image", block_size_h=0)
    with pytest.raises(ValueError, match="block_size_w must be a positive integer"):
        PixelatorOp(fragment=fragment, tensor_name="image", block_size_w=-1)


def test_pixelator_op_missing_tensor_name(fragment):
    """Test PixelatorOp raises error if tensor_name is missing in constructor."""
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'tensor_name'"):
        PixelatorOp(fragment=fragment)


def test_pixelator_op_missing_tensor_name_in_input(
    fragment, op_input_factory, op_output, execution_context, mock_image
):
    """Test PixelatorOp raises error if tensor_name is missing in input message."""
    op = PixelatorOp(fragment=fragment, tensor_name="image")
    op_input = op_input_factory(mock_image((32, 32, 3)), tensor_name="another_image", port="in")
    with pytest.raises(KeyError, match="Tensor 'image' not found in input message"):
        op.compute(op_input, op_output, execution_context)


@pytest.mark.parametrize(
    "expected_shape,block_size",
    [
        ((32, 32, 3), 8),
        ((16, 16, 1), 8),
        ((17, 17, 3), 4),  # Odd shape, block size not divisor
        ((10, 10, 1), 5),
    ],
)
def test_pixelator_op_compute(
    fragment, op_input_factory, op_output, execution_context, mock_image, expected_shape, block_size
):
    """Test PixelatorOp compute: output shape, pixelation, dtype, and ports."""
    tensor_name = "image"
    image = mock_image(expected_shape)
    op_input = op_input_factory(image, tensor_name=tensor_name, port="in")
    op = PixelatorOp(
        block_size_h=block_size,
        block_size_w=block_size,
        tensor_name=tensor_name,
        fragment=fragment,
        name="pixelator_op",
    )
    op.compute(op_input, op_output, execution_context)
    out_msg, out_port = op_output.emitted

    assert out_port == "out", "Output port should be 'out'"
    assert tensor_name in out_msg, f"Output tensor '{tensor_name}' should be in message"
    assert out_msg[tensor_name].shape == expected_shape, f"Output shape should be {expected_shape}"
    assert out_msg[tensor_name].dtype.name == "uint8", "Output dtype should be uint8"

    # Check pixelation: block values should be constant within each block
    for i in range(0, expected_shape[0], block_size):
        for j in range(0, expected_shape[1], block_size):
            block = out_msg[tensor_name][i : i + block_size, j : j + block_size]
            if block.shape[-1] == 1:
                assert (
                    block == block[0, 0, 0]
                ).all(), f"Block at ({i},{j}) not constant (grayscale)"
            else:
                assert (block == block[0, 0, :]).all(), f"Block at ({i},{j}) not constant (color)"
