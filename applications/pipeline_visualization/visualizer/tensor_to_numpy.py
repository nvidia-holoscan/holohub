"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""

import numpy as np
import pipeline_visualization.flatbuffers.DLDataTypeCode
import pipeline_visualization.flatbuffers.IOType
import pipeline_visualization.flatbuffers.Payload
import pipeline_visualization.flatbuffers.Tensor


def tensor_to_numpy(payload: pipeline_visualization.flatbuffers.Payload):
    """Convert a FlatBuffers Payload containing tensor data to a NumPy array.

    Args:
        payload: A FlatBuffers Payload object containing serialized tensor data.

    Returns:
        numpy.ndarray: A NumPy array view of the tensor data with appropriate dtype.

    Raises:
        ValueError: If the tensor's DLDataTypeCode is not supported.
    """
    # Initialize a Tensor object from the FlatBuffers payload
    tensor = pipeline_visualization.flatbuffers.Tensor.Tensor()
    tensor.Init(payload.Bytes, payload.Pos)

    # Extract data type information from the tensor's DLDataType
    # The DLDataType describes the element type using three components:
    code = tensor.Dtype().Code()  # Type category (int, uint, float, etc.)
    bits = tensor.Dtype().Bits()  # Number of bits per element (e.g., 32 for int32)
    lanes = tensor.Dtype().Lanes()  # Number of lanes for vector types (1 for scalars)

    # Map DLDataTypeCode to NumPy dtype kind character
    # 'i' = signed integer, 'u' = unsigned integer, 'f' = floating point
    if code == pipeline_visualization.flatbuffers.DLDataTypeCode.DLDataTypeCode.kDLInt:
        kind = "i"
    elif code == pipeline_visualization.flatbuffers.DLDataTypeCode.DLDataTypeCode.kDLUInt:
        kind = "u"
    elif code == pipeline_visualization.flatbuffers.DLDataTypeCode.DLDataTypeCode.kDLFloat:
        kind = "f"
    else:
        raise ValueError(f"Unknown DLDataTypeCode: {code}")

    # Compose NumPy dtype string, e.g., '<i4' for int32 little-endian
    # Format: '<' (little-endian) + kind (i/u/f) + byte_size
    # FlatBuffers uses little-endian byte order by default
    if bits % 8 != 0:
        raise ValueError(f"Unsupported bit width: {bits} (must be divisible by 8)")
    dtype_str = f"<{kind}{bits // 8}"

    # For vector types (lanes > 1), create a structured dtype with multiple lanes
    # Otherwise, use a simple scalar dtype
    dtype = np.dtype((dtype_str, (lanes,))) if lanes > 1 else np.dtype(dtype_str)

    # Create a NumPy array view of the tensor data with the computed dtype
    numpy_array = tensor.DataAsNumpy().view(dtype)

    # Apply shape to restore original tensor dimensions
    shape_vector = tensor.ShapeAsNumpy()
    if not tensor.ShapeIsNone() and shape_vector is not None and shape_vector.size > 0:
        # the incoming tensor always is in HWC format, so we need to remove the last dimension
        # if it is 1 to match the way numpy arrays are represented
        if shape_vector[-1] == 1:
            shape_vector = shape_vector[:-1]
        shape = tuple(int(dim) for dim in shape_vector)
        strides_vector = tensor.StridesAsNumpy()
        if not tensor.StridesIsNone() and strides_vector is not None and strides_vector.size > 0:
            # If strides_vector has more elements than shape_vector, truncate as needed.
            if strides_vector.size > shape_vector.size:
                strides_vector = strides_vector[: shape_vector.size]
            element_strides = tuple(int(s) for s in strides_vector)
            byte_strides = tuple(stride * numpy_array.dtype.itemsize for stride in element_strides)
            numpy_array = np.lib.stride_tricks.as_strided(
                numpy_array, shape=shape, strides=byte_strides, writeable=False
            )
        else:
            numpy_array = numpy_array.reshape(shape)

    return numpy_array
