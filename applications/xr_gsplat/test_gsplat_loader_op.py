# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", minversion="2.10.0")
pytest.importorskip("holoscan.core")

GsplatLoaderOp = importlib.import_module("applications.xr_gsplat.gsplat_loader_op").GsplatLoaderOp


class _CheckpointPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


@pytest.fixture
def load_checkpoint(monkeypatch, fragment):
    if not torch.cuda.is_available():
        # Keep deserialization and tensor operations real on CPU-only test hosts.
        def on_cpu(function, keyword):
            def call(*args, **kwargs):
                kwargs[keyword] = "cpu"
                return function(*args, **kwargs)

            return call

        for name in ("rand", "zeros", "ones"):
            monkeypatch.setattr(torch, name, on_cpu(getattr(torch, name), "device"))
        monkeypatch.setattr(torch, "load", on_cpu(torch.load, "map_location"))

    return lambda paths: GsplatLoaderOp(fragment, ckpt_paths=paths)


@pytest.mark.parametrize("zip_format", [True, False])
def test_loads_and_combines_training_checkpoints(load_checkpoint, tmp_path, op_output, zip_format):
    paths = []
    for step, position in [(6999, [1.0, 2.0, 3.0]), (29999, [4.0, 5.0, 6.0])]:
        # Match gsplat's simple_trainer.py: checkpoint["splats"] is a state_dict.
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.tensor([position])),
                "quats": torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0]])),
                "scales": torch.nn.Parameter(torch.zeros(1, 3)),
                "opacities": torch.nn.Parameter(torch.zeros(1)),
                "sh0": torch.nn.Parameter(torch.full((1, 1, 3), 0.25)),
                "shN": torch.nn.Parameter(torch.full((1, 3, 3), 0.5)),
            }
        )
        path = tmp_path / f"ckpt_{step}_rank0.pt"
        torch.save(
            {"step": step, "scene_id": "scene", "splats": splats.state_dict()},
            path,
            _use_new_zipfile_serialization=zip_format,
        )
        paths.append(path)

    operator = load_checkpoint(paths)
    operator.compute(None, op_output, None)
    splats, port = op_output.emitted

    assert port == "splats"
    expected = {
        "means": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "scales": torch.ones(2, 3),
        "opacities": torch.full((2,), 0.5),
        "colors": torch.tensor([[[0.25] * 3] + [[0.5] * 3] * 3] * 2),
    }
    for name, value in expected.items():
        torch.testing.assert_close(splats[name].cpu(), value)


@pytest.mark.parametrize("zip_format", [True, False])
def test_rejects_pickle_execution(load_checkpoint, monkeypatch, tmp_path, zip_format):
    # Exercise the old unsafe default on modern PyTorch without overriding an
    # explicit weights_only=True at the application's call site.
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    marker = tmp_path / "pickle-executed"
    checkpoint = tmp_path / "untrusted.pt"
    torch.save(_CheckpointPayload(marker), checkpoint, _use_new_zipfile_serialization=zip_format)

    try:
        with pytest.raises(pickle.UnpicklingError):
            load_checkpoint([checkpoint])
    finally:
        assert not marker.exists(), "Checkpoint deserialization executed the pickle payload"
