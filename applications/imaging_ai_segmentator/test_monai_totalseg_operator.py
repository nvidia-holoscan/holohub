# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch", minversion="2.6.0")
monai = pytest.importorskip("monai")
pytest.importorskip("holoscan.core")


class _CheckpointPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


@pytest.fixture
def total_seg(monkeypatch):
    # Isolate DICOM I/O and downstream inference. Loading the source module keeps
    # the real Holoscan operator, MONAI network and PyTorch deserialization.
    core = ModuleType("operators.medical_imaging.core")
    core.AppContext = SimpleNamespace
    core.Model = lambda *args, **kwargs: SimpleNamespace()
    inference = ModuleType("operators.medical_imaging.monai_seg_inference_operator")
    inference.InfererType = SimpleNamespace(SLIDING_WINDOW="sliding_window")
    inference.InMemImageReader = lambda image: image
    inference.MonaiSegInferenceOperator = lambda *args, **kwargs: SimpleNamespace(
        compute_impl=lambda image, context: image
    )
    monkeypatch.setitem(sys.modules, core.__name__, core)
    monkeypatch.setitem(sys.modules, inference.__name__, inference)
    spec = importlib.util.spec_from_file_location(
        "monai_totalseg_operator", Path(__file__).with_name("monai_totalseg_operator.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_checkpoint(total_seg, monkeypatch, fragment, tmp_path, op_output):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(total_seg.MonaiTotalSegOperator, "pre_process", lambda *args: None)
    monkeypatch.setattr(total_seg.MonaiTotalSegOperator, "post_process", lambda *args: None)

    def load(path):
        app_context = SimpleNamespace(models=None)
        operator = total_seg.MonaiTotalSegOperator(
            fragment,
            app_context=app_context,
            model_path=path,
            output_folder=tmp_path / "output",
        )
        operator.compute(SimpleNamespace(receive=lambda port: object()), op_output, None)
        return app_context.models.predictor

    return load


@pytest.mark.parametrize("zip_format", [True, False])
def test_loads_segresnet_state_dict(load_checkpoint, tmp_path, zip_format):
    model = monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )
    expected = model.state_dict()
    checkpoint = tmp_path / "model.pt"
    torch.save(expected, checkpoint, _use_new_zipfile_serialization=zip_format)

    loaded = load_checkpoint(checkpoint)

    torch.testing.assert_close(loaded.state_dict(), expected)
    assert not loaded.training


@pytest.mark.parametrize("zip_format", [True, False])
def test_rejects_pickle_execution(load_checkpoint, monkeypatch, tmp_path, zip_format):
    # Reproduce the pre-2.6 default even when running a newer PyTorch release.
    # This override applies only when the caller omits weights_only.
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    marker = tmp_path / "pickle-executed"
    checkpoint = tmp_path / "untrusted.pt"
    torch.save(_CheckpointPayload(marker), checkpoint, _use_new_zipfile_serialization=zip_format)

    try:
        with pytest.raises(pickle.UnpicklingError):
            load_checkpoint(checkpoint)
    finally:
        assert not marker.exists(), "Checkpoint deserialization executed the pickle payload"
