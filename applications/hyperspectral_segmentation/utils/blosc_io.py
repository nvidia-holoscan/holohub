# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the numeric array subset of the HeiPorSPECTRAL Blosc format."""

import io
import math
import pickle
import pickletools
from pathlib import Path

import blosc
import numpy as np


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Existing dataset headers use NumPy's dtype constructor. No other
        # globals are needed to describe an array's shape and scalar dtype.
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"Unsupported metadata global: {module}.{name}")


def _read_metadata(payload):
    try:
        # Inspect and unpickle the same immutable bytes. EXT opcodes can reuse
        # cached globals without calling find_class, so reject them explicitly.
        for opcode, _, position in pickletools.genops(payload):
            if opcode.name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"}:
                raise ValueError("Unsupported metadata opcode")
            if opcode.name == "STOP":
                header_end = position + 1
                metadata = _MetadataUnpickler(io.BytesIO(payload[:header_end])).load()
                return metadata, header_end
    except Exception as exc:
        # Malformed dtype arguments can raise exceptions beyond pickle's own
        # errors. Normalize failures only within this metadata decoding boundary.
        raise ValueError("Invalid Blosc array metadata") from exc
    raise ValueError("Missing Blosc array metadata")


def _decompress_array(data, shape, dtype):
    if not isinstance(shape, tuple) or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Array shape must contain nonnegative integers")
    if (
        not isinstance(dtype, np.dtype)
        or dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.kind not in "buifc"
    ):
        raise ValueError("Only scalar numeric array dtypes are supported")
    expected_size = math.prod(shape) * dtype.itemsize
    if not blosc.cbuffer_validate(data):
        raise ValueError("Invalid compressed Blosc buffer")
    uncompressed_size, compressed_size, _ = blosc.get_cbuffer_sizes(data)
    if uncompressed_size != expected_size or compressed_size != len(data):
        raise ValueError("Compressed buffer sizes do not match array metadata")
    data = blosc.decompress(data)
    if len(data) != expected_size:
        raise ValueError("Decompressed data size does not match array metadata")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def decompress_file(path):
    """Load one numeric array, or a named collection, without arbitrary unpickling."""
    payload = Path(path).read_bytes()
    metadata, offset = _read_metadata(payload)
    if isinstance(metadata, tuple) and len(metadata) == 2:
        return _decompress_array(payload[offset:], *metadata)
    if isinstance(metadata, dict):
        result = {}
        for name, descriptor in metadata.items():
            if (
                not isinstance(name, str)
                or not isinstance(descriptor, tuple)
                or len(descriptor) != 3
            ):
                raise ValueError("Invalid named array metadata")
            shape, dtype, size = descriptor
            if type(size) is not int or size < 0 or size > len(payload) - offset:
                raise ValueError("Invalid compressed array size")
            result[name] = _decompress_array(payload[offset : offset + size], shape, dtype)
            offset += size
        if offset != len(payload):
            raise ValueError("Unexpected data after named arrays")
        return result
    raise ValueError("Invalid Blosc array metadata")
