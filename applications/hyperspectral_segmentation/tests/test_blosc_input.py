# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copyreg
import importlib
import pickle
from pathlib import Path

import blosc
import numpy as np
import pytest


@pytest.fixture
def load_cube(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("hyperspectral_segmentation")
    return lambda path: module.LoadDataOp.decompress_file(None, path)


def write_cube(path, array, metadata=None, protocol=4):
    with path.open("wb") as stream:
        pickle.dump((array.shape, array.dtype) if metadata is None else metadata, stream, protocol)
        stream.write(blosc.compress(array.tobytes(), typesize=array.dtype.itemsize))


def test_pickle_payload_is_rejected_without_execution(load_cube, tmp_path):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    path = tmp_path / "payload.blosc"
    path.write_bytes(pickle.dumps(Payload()))
    try:
        with pytest.raises(ValueError):
            load_cube(path)
    finally:
        assert not marker.exists()


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("dtype", ["float16", ">f4", "int16"])
def test_legacy_numeric_cube_round_trip(load_cube, tmp_path, protocol, dtype):
    expected = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    path = tmp_path / "cube.blosc"
    write_cube(path, expected, protocol=protocol)
    actual = load_cube(path)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.flags.writeable


@pytest.mark.parametrize(
    "metadata",
    [
        ((1,), np.dtype(object)),
        ((1,), np.dtype([("field", "f4")])),
        ((1,), np.dtype(("f4", (2,)))),
        ((True,), np.dtype("f4")),
        ((-1,), np.dtype("f4")),
        ((1.0,), np.dtype("f4")),
    ],
)
def test_invalid_metadata_is_rejected(load_cube, tmp_path, metadata):
    path = tmp_path / "invalid.blosc"
    write_cube(path, np.array([], dtype=np.float32), metadata=metadata)
    with pytest.raises(ValueError):
        load_cube(path)


def test_decompressed_size_must_match_shape(load_cube, tmp_path):
    path = tmp_path / "short.blosc"
    write_cube(path, np.array([1.0], dtype=np.float32), metadata=((2,), np.dtype("f4")))
    with pytest.raises(ValueError):
        load_cube(path)


def test_size_mismatch_is_rejected_before_decompression(load_cube, tmp_path, monkeypatch):
    path = tmp_path / "oversized.blosc"
    write_cube(path, np.zeros(1024, dtype=np.float32), metadata=((1,), np.dtype("f4")))

    def unexpected_decompression(data):
        pytest.fail("Mismatched buffer reached decompression")

    monkeypatch.setattr(blosc, "decompress", unexpected_decompression)
    with pytest.raises(ValueError):
        load_cube(path)


@pytest.mark.parametrize("code", [123, 12345, 23456789])
def test_cached_pickle_extension_cannot_bypass_global_restrictions(load_cube, tmp_path, code):
    marker = tmp_path / "executed"
    copyreg.add_extension("builtins", "eval", code)
    try:
        # Simulate another component having already cached a global for EXT4.
        assert pickle.loads(pickle.dumps(eval, protocol=2)) is eval

        class Payload:
            def __reduce__(self):
                return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

        path = tmp_path / "extension.blosc"
        path.write_bytes(pickle.dumps(Payload(), protocol=2))
        try:
            with pytest.raises(ValueError):
                load_cube(path)
        finally:
            assert not marker.exists()
    finally:
        copyreg.remove_extension("builtins", "eval", code)
        copyreg.clear_extension_cache()


def test_named_arrays_remain_supported(load_cube, tmp_path):
    arrays = {
        "cube": np.arange(24, dtype=np.float16).reshape(2, 3, 4),
        "labels": np.array([1, 2], dtype=np.int16),
    }
    blocks = {
        name: blosc.compress(array.tobytes(), typesize=array.dtype.itemsize)
        for name, array in arrays.items()
    }
    metadata = {
        name: (array.shape, array.dtype, len(blocks[name])) for name, array in arrays.items()
    }
    path = tmp_path / "named.blosc"
    path.write_bytes(pickle.dumps(metadata) + b"".join(blocks.values()))
    actual = load_cube(path)
    assert actual.keys() == arrays.keys()
    for name, expected in arrays.items():
        np.testing.assert_array_equal(actual[name], expected)


@pytest.mark.parametrize("size", [-1, True, 100000])
def test_invalid_named_array_lengths_are_rejected(load_cube, tmp_path, size):
    path = tmp_path / "invalid.blosc"
    write_cube(path, np.array([1], dtype=np.int16), metadata={"cube": ((1,), np.dtype("i2"), size)})
    with pytest.raises(ValueError):
        load_cube(path)


@pytest.mark.parametrize("payload", [b"", b"not a pickle", pickle.dumps((1,))])
def test_malformed_headers_are_rejected(load_cube, tmp_path, payload):
    path = tmp_path / "invalid.blosc"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_cube(path)
