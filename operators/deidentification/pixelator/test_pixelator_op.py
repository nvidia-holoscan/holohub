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
from holoscan.core import Operator, _Operator

from .pixelator_op import PixelatorOp


def test_pixelator_op_init(fragment):
    name = "pixelator_op"
    op = PixelatorOp(fragment=fragment, name=name)

    # Check if it is a Holoscan operator
    assert isinstance(op, _Operator)

    # Check operator type
    assert op.operator_type == Operator.OperatorType.NATIVE

    # Check operator name is matched
    assert f"name: {name}" in repr(op)


def test_pixelator_op_setup(fragment):
    op = PixelatorOp(fragment=fragment)
    spec = op.spec

    # Check input and output ports
    assert "in" in spec.inputs
    assert "out" in spec.outputs


@pytest.mark.parametrize("expected_shape", [(32, 32, 3), (16, 16, 1)])
def test_pixelator_op_compute(
    fragment, op_input_factory, op_output, context, dummy_image_factory, expected_shape
):
    tensor_name = "image"
    image = dummy_image_factory(expected_shape)
    op_input = op_input_factory(image, tensor_name=tensor_name, port="in")
    op = PixelatorOp(
        block_size_h=8,
        block_size_w=8,
        tensor_name=tensor_name,
        fragment=fragment,
        name="pixelator_op",
    )
    op.compute(op_input, op_output, context)
    out_msg, out_port = op_output.emitted

    # Check output port
    assert out_port == "out"

    # Check output tensor name
    assert tensor_name in out_msg

    # Check output tensor shape
    assert out_msg[tensor_name].shape == expected_shape

    # Check pixelation effect: block values should be constant within 8x8 blocks
    for i in range(0, expected_shape[0], 8):
        for j in range(0, expected_shape[1], 8):
            block = out_msg[tensor_name][i : i + 8, j : j + 8]
            # For grayscale, block has shape (8,8,1); for RGB, (8,8,3)
            if block.shape[-1] == 1:
                assert (block == block[0, 0, 0]).all()
            else:
                assert (block == block[0, 0, :]).all()
