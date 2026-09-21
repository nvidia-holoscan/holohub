# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from holoscan.core import Fragment


@pytest.fixture
def operator(tmp_path, monkeypatch):
    # Bypass unrelated eager DICOM imports while exercising the real operator,
    # AppContext, and Image implementations.
    package_path = Path(__file__).resolve().parents[2] / "operators" / "medical_imaging"
    existing_modules = set(sys.modules)
    for name, path in (
        ("operators.medical_imaging", package_path),
        ("operators.medical_imaging.core.domain", package_path / "core" / "domain"),
    ):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    core = importlib.import_module("operators.medical_imaging.core")
    module = importlib.import_module("operators.medical_imaging.monai_bundle_inference_operator")
    op = module.MonaiBundleInferenceOperator(
        Fragment(),
        app_context=core.AppContext({"model": str(tmp_path / "missing-model")}),
        input_mapping=[],
        output_mapping=[],
    )
    op._inputs = {"data": {"type": "series"}}
    op._outputs = {"data": {"type": "series"}}
    op._device = "cpu"
    yield op
    for name in set(sys.modules) - existing_modules:
        if name.startswith("operators.medical_imaging."):
            sys.modules.pop(name, None)


def receive(operator, value):
    return operator._receive_input("data", SimpleNamespace(receive=lambda name: value), None)


@pytest.mark.parametrize(
    "input_type, file_format", [(object, "pickle"), ("series", "pickle"), ("series", "npy")]
)
def test_pickle_payload_is_rejected_without_execution(operator, tmp_path, input_type, file_format):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    path = tmp_path / "data"
    with path.open("wb") as stream:
        if file_format == "npy":
            np.save(stream, np.array([Payload()], dtype=object))
        else:
            pickle.dump(Payload(), stream)
    operator._inputs["data"]["type"] = input_type

    try:
        with pytest.raises(ValueError):
            receive(operator, path)
    finally:
        assert not marker.exists()


@pytest.mark.parametrize("location", ["file", "named-directory", "single-file-directory"])
def test_numeric_disk_output_round_trip(operator, tmp_path, location):
    expected = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    output = SimpleNamespace(get=lambda name: tmp_path)
    operator._send_output(expected, "data", {}, output, None)
    path = tmp_path / "data"
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)

    if location == "single-file-directory":
        path.rename(tmp_path / "array.npy")
    value, metadata = receive(operator, path if location == "file" else tmp_path)

    torch.testing.assert_close(value, torch.tensor([[1.5, 2.5], [3.5, 4.5]]))
    assert metadata is None
    assert not (tmp_path / "data.npy").exists()


def test_archive_input_is_rejected(operator, tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, data=np.array([1.0], dtype=np.float32))
    with pytest.raises(TypeError):
        receive(operator, path)


def test_unsupported_disk_output_is_rejected(operator, tmp_path):
    operator._outputs["data"]["type"] = "image"
    output = SimpleNamespace(get=lambda name: tmp_path, emit=lambda *args: None)
    with pytest.raises(ValueError):
        operator._send_output(torch.ones((1, 2, 3, 4)), "data", {}, output, None)
    assert not (tmp_path / "data").exists()


def test_in_memory_inputs_remain_supported(operator):
    value, metadata = receive(operator, np.array([1, 2, 3], dtype=np.int32))
    torch.testing.assert_close(value, torch.tensor([1, 2, 3], dtype=torch.int32))
    assert metadata is None
    original = {"labels": ["spleen"], "probabilities": [0.75]}
    operator._inputs["data"]["type"] = "probabilities"
    assert receive(operator, original) == (original, None)


def test_in_memory_image_preserves_pixels_and_metadata(operator):
    from operators.medical_imaging.core import Image

    pixels = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    metadata = {"spacing": [1, 2, 3], "original_affine": np.eye(4)}
    operator._inputs["data"]["type"] = "image"
    value, actual_metadata = receive(operator, Image(pixels, metadata))
    np.testing.assert_array_equal(value, pixels)
    assert actual_metadata is metadata


@pytest.mark.parametrize("get_raises", [False, True])
def test_in_memory_output_is_emitted(operator, get_raises):
    expected = np.array([1, 2, 3], dtype=np.float32)
    emitted = []

    def get(name):
        if get_raises:
            raise RuntimeError("In-memory port has no disk output")

    output = SimpleNamespace(get=get, emit=lambda value, name: emitted.append((value, name)))
    operator._send_output(expected, "data", {}, output, None)
    assert len(emitted) == 1
    np.testing.assert_array_equal(emitted[0][0], expected)
    assert emitted[0][1] == "data"


def test_object_array_output_is_rejected_before_writing(operator, tmp_path):
    output = SimpleNamespace(
        get=lambda name: tmp_path, emit=lambda *args: pytest.fail("Emitted invalid disk output")
    )
    with pytest.raises(ValueError):
        operator._send_output(np.array([object()]), "data", {}, output, None)
    assert not (tmp_path / "data").exists()


def test_disk_write_failure_is_not_swallowed(operator, tmp_path):
    (tmp_path / "data").mkdir()
    output = SimpleNamespace(
        get=lambda name: tmp_path, emit=lambda *args: pytest.fail("Swallowed disk write failure")
    )
    with pytest.raises(IsADirectoryError):
        operator._send_output(np.array([1]), "data", {}, output, None)
