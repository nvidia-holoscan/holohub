# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

"""HoloViz overlay tests for procedural_narrator."""

from collections import defaultdict

import numpy as np
import pytest

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


def test_application_rejects_unknown_source():
    with pytest.raises(ValueError, match="source"):
        ProceduralNarratorApp("/tmp/data", source="file")


def test_event_sink_ignores_an_empty_receive():
    class State:
        events = []

        def apply_event(self, event):
            self.events.append(event)

    class Input:
        def receive(self, port):
            assert port == "events"
            return None

    state = State()
    sink = NarrativeEventSinkOp(Application(), state=state)

    sink.compute(Input(), None, None)

    assert state.events == []


def test_compose_uses_reasoner_default_clip_duration_when_omitted(monkeypatch):
    captured_clip_durations = []

    def stub_operator(*args, **kwargs):
        return object()

    class Reasoner:
        def __init__(self, *args, **kwargs):
            self.clip_duration_s = kwargs.get("clip_duration_s", 7.5)

    class State:
        def __init__(self, clip_duration_s):
            captured_clip_durations.append(clip_duration_s)

    class App:
        _endpoint = None
        _headless = True

        def kwargs(self, name):
            return {
                "format_converter": {"resize_width": 640, "resize_height": 480},
                "reasoner": {},
                "holoviz": {},
            }[name]

        def _make_source(self, allocator):
            return object(), "output", "rgb888"

        def add_flow(self, *args, **kwargs):
            pass

        def on_window_closed(self):
            pass

    for name in (
        "UnboundedAllocator",
        "CudaStreamPool",
        "FormatConverterOp",
        "NarrativeEventSinkOp",
        "NarrativeDisplayOp",
        "HolovizOp",
        "PeriodicCondition",
    ):
        monkeypatch.setattr(narrator_module, name, stub_operator)
    monkeypatch.setattr(narrator_module, "OnlineVideoReasonerOp", Reasoner)
    monkeypatch.setattr(narrator_module, "NarrativeState", State)

    ProceduralNarratorApp.compose(App())

    assert captured_clip_durations == [7.5]


def test_display_repaints_no_signal_without_a_new_frame(monkeypatch):
    state = narrator_module.NarrativeState(clip_duration_s=4.0, input_timeout_s=1.0)
    display = NarrativeDisplayOp(
        Application(),
        state=state,
        frame_width=4,
        frame_height=3,
    )
    source_frame = np.full((3, 4, 3), 17, dtype=np.uint8)

    class Input:
        messages = iter(({"frame": source_frame}, None))

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
