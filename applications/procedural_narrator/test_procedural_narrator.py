# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

"""HoloViz overlay tests for procedural_narrator."""

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("holoscan")

from holoscan.core import Application

import applications.procedural_narrator.procedural_narrator as narrator_module
from applications.procedural_narrator.narrative_state import NarrativeSnapshot
from applications.procedural_narrator.procedural_narrator import (
    NarrativeDisplayOp,
    NarrativeEventSinkOp,
    ProceduralNarratorApp,
    build_overlay,
    working_pulse_alpha,
)


def test_overlay_contains_bottom_card_and_compact_status_circles():
    snapshot = NarrativeSnapshot(
        input_live=True,
        input_label="VIDEO · LIVE",
        model_phase="waiting",
        model_label="MODEL · WAITING",
        narrative="The endoscope advances while a grasper moves toward tissue.",
        working_message="Temporal observation submitted. Waiting for the model.",
    )

    tensors, specs = build_overlay(snapshot, blink_on=True, working_alpha=0.55)

    assert {
        "narrative_card",
        "narrative_heading",
        "narrative_text",
        "video_status_outline",
        "video_status_fill",
        "model_status_outline",
        "model_status_fill",
        "working_text",
    } == set(tensors)
    assert all(value.dtype == np.float32 for value in tensors.values())
    texts = [text for spec in specs for text in spec.text]
    assert "CLINICAL NARRATIVE" in texts
    assert "VIDEO · LIVE" not in texts
    assert "MODEL · WAITING" not in texts
    assert tensors["narrative_card"][:, 1].min() > 0.7
    assert specs[4].color == pytest.approx(narrator_module.GREEN)
    assert specs[6].color == pytest.approx(narrator_module.AMBER)
    assert specs[2].color == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert specs[7].color == pytest.approx([*narrator_module.DIM_TEXT_COLOR[:3], 0.55])
    assert tensors["working_text"][0, 1] > tensors["narrative_text"][-1, 1]


def test_overlay_uses_hollow_ready_indicator_and_bounded_narrative_lines():
    snapshot = NarrativeSnapshot(
        input_live=True,
        input_label="VIDEO · LIVE",
        model_phase="ready",
        model_label="MODEL · READY",
        narrative="word " * 200,
        working_message="Collecting temporal context for the next procedural narrative.",
    )

    tensors, specs = build_overlay(snapshot, blink_on=True)

    assert tensors["narrative_text"].shape[0] == 3
    texts = [text for spec in specs for text in spec.text]
    assert any(text.endswith("…") for text in texts)
    assert specs[6].color == narrator_module.TRANSPARENT


def test_overlay_preserves_the_tail_of_streamed_narration():
    snapshot = NarrativeSnapshot(
        input_live=True,
        input_label="VIDEO · LIVE",
        model_phase="narrating",
        model_label="MODEL · NARRATING",
        narrative=" ".join(f"word{index}" for index in range(100)),
        working_message="Receiving the current procedural narrative.",
    )

    _, specs = build_overlay(snapshot, blink_on=True)

    assert specs[2].text[-1].endswith("word99")
    assert specs[2].text[0].startswith("…")


def test_model_indicator_blinks_while_waiting_and_is_green_while_narrating():
    waiting = NarrativeSnapshot(
        input_live=True,
        input_label="VIDEO · LIVE",
        model_phase="waiting",
        model_label="MODEL · WAITING",
        narrative="Waiting.",
        working_message="Temporal observation submitted. Waiting for the model.",
    )
    narrating = NarrativeSnapshot(
        input_live=True,
        input_label="VIDEO · LIVE",
        model_phase="narrating",
        model_label="MODEL · NARRATING",
        narrative="A grasper moves.",
        working_message="Receiving the current procedural narrative.",
    )

    _, waiting_off_specs = build_overlay(waiting, blink_on=False)
    _, narrating_specs = build_overlay(narrating, blink_on=False)

    assert waiting_off_specs[6].color == narrator_module.TRANSPARENT
    assert narrating_specs[6].color == pytest.approx(narrator_module.GREEN)


def test_working_message_pulses_without_dimming_narration():
    snapshot = NarrativeSnapshot(
        input_live=True,
        input_label="VIDEO · LIVE",
        model_phase="collecting",
        model_label="MODEL · COLLECTING 50%",
        narrative="The endoscope advances.",
        working_message="Collecting temporal context.",
    )

    _, dim_specs = build_overlay(snapshot, blink_on=False, working_alpha=0.35)
    _, bright_specs = build_overlay(snapshot, blink_on=False, working_alpha=0.75)

    assert dim_specs[2].color == bright_specs[2].color == narrator_module.TEXT_COLOR
    assert dim_specs[7].color[:3] == bright_specs[7].color[:3]
    assert dim_specs[7].color[3] == pytest.approx(0.35)
    assert bright_specs[7].color[3] == pytest.approx(0.75)
    assert working_pulse_alpha(0.75) == pytest.approx(narrator_module.WORKING_PULSE_MAX_ALPHA)
    assert working_pulse_alpha(2.25) == pytest.approx(narrator_module.WORKING_PULSE_MIN_ALPHA)


def test_application_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="source"):
        ProceduralNarratorApp(str(tmp_path), source="file")


def test_default_replayer_configuration_has_explicit_cadence_and_no_loop():
    config_path = Path(narrator_module.__file__).with_name("procedural_narrator.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["replayer_source"]["frame_rate"] == 30
    assert config["replayer_source"]["realtime"] is True
    assert config["replayer_source"]["repeat"] is False


def test_replayer_settings_return_the_configured_cadence():
    assert narrator_module._validate_replayer_settings(
        {"frame_rate": 15, "repeat": False}
    ) == pytest.approx(15.0)


@pytest.mark.parametrize("frame_rate", [0, -1, float("nan"), float("inf"), True, "30"])
def test_replayer_settings_require_a_positive_explicit_cadence(frame_rate):
    with pytest.raises(ValueError, match="frame_rate must be a positive finite number"):
        narrator_module._validate_replayer_settings({"frame_rate": frame_rate, "repeat": False})


@pytest.mark.parametrize("repeat", [True, 1, "false"])
def test_replayer_settings_reject_repetition(repeat):
    with pytest.raises(ValueError, match="repeat must be false"):
        narrator_module._validate_replayer_settings({"frame_rate": 30, "repeat": repeat})


@pytest.mark.parametrize("realtime", [False, 0, "true"])
def test_replayer_settings_require_realtime_playback(realtime):
    with pytest.raises(ValueError, match="realtime must be true"):
        narrator_module._validate_replayer_settings(
            {"frame_rate": 30, "realtime": realtime, "repeat": False}
        )


def test_event_sink_ignores_an_empty_receive():
    class State:
        def __init__(self):
            self.events = []

        def apply_event(self, event):
            self.events.append(event)

    class Input:
        def receive(self, port):
            assert port == "events"

    state = State()
    sink = NarrativeEventSinkOp(Application(), state=state)

    sink.compute(Input(), None, None)

    assert state.events == []


def test_compose_wires_reasoner_state_and_validates_source_sample_rate(monkeypatch):
    captured_state_args = []
    captured_reasoner_tensor_names = []
    captured_display_tensor_names = []

    def stub_operator(*args, **kwargs):
        return object()

    def stub_display(*args, **kwargs):
        captured_display_tensor_names.append(kwargs["input_tensor_name"])
        return object()

    class Reasoner:
        def __init__(self, *args, **kwargs):
            self.clip_duration_s = kwargs.get("clip_duration_s", 7.5)
            self.max_frame_gap_s = kwargs.get("max_frame_gap_s", 0.25)
            self.sample_fps = kwargs.get("sample_fps", 4.0)
            self.request_options = dict(kwargs.get("request_options", {}))
            captured_reasoner_tensor_names.append(kwargs["tensor_name"])

    class State:
        def __init__(self, clip_duration_s, max_frame_gap_s, thinking_mode):
            captured_state_args.append((clip_duration_s, max_frame_gap_s, thinking_mode))

    class App:
        _endpoint = None
        _headless = True

        def __init__(self, sample_fps=15.0, source="replayer", source_frame_rate=15.0):
            self.sample_fps = sample_fps
            self._source = source
            self.source_frame_rate = source_frame_rate

        def kwargs(self, name):
            return {
                "format_converter": {
                    "out_tensor_name": "converted_video",
                    "resize_width": 640,
                    "resize_height": 480,
                },
                "reasoner": {
                    "sample_fps": self.sample_fps,
                    "request_options": {
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                        }
                    },
                },
                "holoviz": {},
            }[name]

        def _make_source(self, allocator):
            source_frame_rate = self.source_frame_rate if self._source == "replayer" else None
            return object(), "output", "rgb888", source_frame_rate

        def add_flow(self, *args, **kwargs):
            pass

        def on_window_closed(self):
            pass

    for name in (
        "UnboundedAllocator",
        "CudaStreamPool",
        "FormatConverterOp",
        "NarrativeEventSinkOp",
        "HolovizOp",
        "PeriodicCondition",
    ):
        monkeypatch.setattr(narrator_module, name, stub_operator)
    monkeypatch.setattr(narrator_module, "NarrativeDisplayOp", stub_display)
    monkeypatch.setattr(narrator_module, "OnlineVideoReasonerOp", Reasoner)
    monkeypatch.setattr(narrator_module, "NarrativeState", State)

    ProceduralNarratorApp.compose(App())

    assert captured_state_args == [(7.5, 0.25, True)]
    assert captured_reasoner_tensor_names == ["converted_video"]
    assert captured_display_tensor_names == ["converted_video"]

    with pytest.raises(ValueError, match=r"replayer_source\.frame_rate \(15 fps\)"):
        ProceduralNarratorApp.compose(App(sample_fps=15.01, source_frame_rate=15.0))

    ProceduralNarratorApp.compose(App(sample_fps=60.0, source="v4l2"))

    with pytest.raises(ValueError, match="60 Hz reasoner tick rate"):
        ProceduralNarratorApp.compose(App(sample_fps=60.01, source="v4l2"))


def test_display_rejects_a_non_string_input_tensor_name():
    with pytest.raises(TypeError, match="input_tensor_name"):
        NarrativeDisplayOp(
            Application(),
            state=narrator_module.NarrativeState(clip_duration_s=4.0),
            frame_width=4,
            frame_height=3,
            input_tensor_name=None,
        )


def test_display_repaints_no_signal_without_a_new_frame(monkeypatch):
    state = narrator_module.NarrativeState(clip_duration_s=4.0, input_timeout_s=1.0)
    display = NarrativeDisplayOp(
        Application(),
        state=state,
        frame_width=4,
        frame_height=3,
        input_tensor_name="converted_video",
    )
    source_frame = np.full((3, 4, 3), 17, dtype=np.uint8)

    class Input:
        messages = iter(({"converted_video": source_frame}, None))

        def receive(self, port):
            assert port == "video"
            return next(self.messages)

    class Output:
        def __init__(self):
            self.values = defaultdict(list)

        def emit(self, value, port):
            self.values[port].append(value)

    times = iter((10.0, 12.0))
    monkeypatch.setattr(narrator_module.time, "monotonic", times.__next__)
    output = Output()

    display.compute(Input(), output, None)
    source_frame.fill(99)
    display.compute(Input(), output, None)

    assert np.all(output.values["tensors"][1]["frame"] == 17)
    assert output.values["specs"][1][4].color == narrator_module.TRANSPARENT


def test_display_synchronises_cuda_default_stream(monkeypatch):
    state = narrator_module.NarrativeState(clip_duration_s=4.0)
    display = NarrativeDisplayOp(
        Application(),
        state=state,
        frame_width=4,
        frame_height=3,
        input_tensor_name="converted_video",
    )
    tensor = SimpleNamespace(__cuda_array_interface__={})
    expected = np.full((3, 4, 3), 23, dtype=np.uint8)
    calls = []

    class StreamContext:
        def __enter__(self):
            calls.append(("enter", 0))

        def __exit__(self, *args):
            calls.append(("exit", 0))

    def external_stream(stream):
        calls.append(("external_stream", stream))
        return StreamContext()

    def asarray(value):
        calls.append(("asarray", value))
        return expected

    def asnumpy(value):
        calls.append(("asnumpy", value))
        return value

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asarray=asarray,
            asnumpy=asnumpy,
            cuda=SimpleNamespace(ExternalStream=external_stream),
        ),
    )
    monkeypatch.setattr(narrator_module.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(narrator_module, "build_overlay", lambda *args, **kwargs: ({}, []))

    class Input:
        def receive(self, port):
            assert port == "video"
            return {"converted_video": tensor}

        def receive_cuda_stream(self, port, *, allocate):
            assert port == "video"
            assert not allocate
            return 0

    class Output:
        def __init__(self):
            self.values = {}

        def emit(self, value, port):
            self.values[port] = value

    output = Output()
    display.compute(Input(), output, None)

    assert calls == [
        ("external_stream", 0),
        ("enter", 0),
        ("asarray", tensor),
        ("asnumpy", expected),
        ("exit", 0),
    ]
    assert output.values["tensors"]["frame"] is expected


def test_display_missing_tensor_error_names_configured_input(monkeypatch):
    display = NarrativeDisplayOp(
        Application(),
        state=narrator_module.NarrativeState(clip_duration_s=4.0),
        frame_width=4,
        frame_height=3,
        input_tensor_name="converted_video",
    )

    class Input:
        def receive(self, port):
            assert port == "video"
            return {"frame": np.zeros((3, 4, 3), dtype=np.uint8)}

    monkeypatch.setattr(narrator_module.time, "monotonic", lambda: 10.0)

    with pytest.raises(ValueError, match="'converted_video'"):
        display.compute(Input(), None, None)
