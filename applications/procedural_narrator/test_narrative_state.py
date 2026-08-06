# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

"""Tests for procedural narrator display state."""

import pytest

from applications.procedural_narrator.narrative_state import NarrativeState, wrap_narrative


def test_state_reports_video_collection_and_staleness():
    state = NarrativeState(clip_duration_s=4.0, input_timeout_s=1.0)

    assert state.snapshot(0.0).input_label == "VIDEO · NO SIGNAL"
    assert state.snapshot(0.0).model_label == "MODEL · IDLE"

    state.note_frame(10.0)
    collecting = state.snapshot(12.0)
    assert collecting.input_label == "VIDEO · NO SIGNAL"
    assert collecting.model_label == "MODEL · COLLECTING 50%"
    assert collecting.narrative == ""
    assert collecting.working_message.startswith("Collecting temporal context")

    state.note_frame(12.0)
    assert state.snapshot(12.5).input_label == "VIDEO · LIVE"
    assert state.snapshot(14.0).input_label == "VIDEO · NO SIGNAL"


def test_state_restarts_collection_timing_after_a_frame_gap():
    state = NarrativeState(
        clip_duration_s=4.0,
        input_timeout_s=1.0,
        max_frame_gap_s=0.5,
    )
    state.note_frame(10.0)
    assert state.snapshot(14.0).model_label == "MODEL · READY"

    state.note_frame(15.0)
    resumed = state.snapshot(15.0)

    assert resumed.input_label == "VIDEO · LIVE"
    assert resumed.model_label == "MODEL · COLLECTING 0%"


def test_state_tracks_waiting_streaming_and_completed_response():
    state = NarrativeState(clip_duration_s=4.0)
    state.note_frame(1.0)

    assert state.apply_event({"request_id": "r1", "kind": "started"})
    waiting = state.snapshot(1.1)
    assert waiting.model_label == "MODEL · WAITING"
    assert waiting.narrative == ""
    assert waiting.working_message == "Temporal observation submitted. Waiting for the model."

    assert state.apply_event({"request_id": "r1", "kind": "delta", "text": "A grasper "})
    assert state.apply_event({"request_id": "r1", "kind": "delta", "text": "moves left."})
    streaming = state.snapshot(1.2)
    assert streaming.model_label == "MODEL · NARRATING"
    assert streaming.narrative == "A grasper moves left."

    assert state.apply_event(
        {
            "request_id": "r1",
            "kind": "completed",
            "text": "A grasper moves left and exposes tissue.",
            "deltas_dropped": False,
        }
    )
    completed = state.snapshot(1.3)
    assert completed.model_label == "MODEL · READY"
    assert completed.narrative == "A grasper moves left and exposes tissue."
    assert completed.working_message.startswith("Collecting temporal context")

    assert state.apply_event({"request_id": "r2", "kind": "started"})
    next_request = state.snapshot(1.4)
    assert next_request.narrative == completed.narrative
    assert next_request.working_message == (
        "Temporal observation submitted. Waiting for the model."
    )


def test_completed_event_replaces_partial_or_dropped_stream():
    state = NarrativeState(clip_duration_s=4.0)
    state.apply_event({"request_id": "r1", "kind": "delta", "text": "Partial"})
    state.apply_event(
        {
            "request_id": "r1",
            "kind": "completed",
            "text": "Canonical completed response.",
            "deltas_dropped": True,
        }
    )

    assert state.snapshot(0.0).narrative == "Canonical completed response."


def test_state_strips_thinking_blocks_from_streamed_and_completed_text():
    state = NarrativeState(clip_duration_s=4.0)
    state.apply_event({"request_id": "r1", "kind": "started"})
    state.apply_event({"request_id": "r1", "kind": "delta", "text": "<"})
    state.apply_event({"request_id": "r1", "kind": "delta", "text": "think>private analysis"})

    assert "private analysis" not in state.snapshot(0.0).narrative

    state.apply_event(
        {
            "request_id": "r1",
            "kind": "delta",
            "text": "</think>\nA grasper moves left.",
        }
    )
    assert state.snapshot(0.0).narrative == "A grasper moves left."

    state.apply_event(
        {
            "request_id": "r1",
            "kind": "completed",
            "text": "<think>private analysis</think>\nThe tissue plane is exposed.",
        }
    )
    assert state.snapshot(0.0).narrative == "The tissue plane is exposed."

    state.apply_event(
        {
            "request_id": "r2",
            "kind": "completed",
            "text": "private analysis</think>Visible narrative.",
        }
    )
    assert state.snapshot(0.0).narrative == "Visible narrative."


def test_thinking_mode_buffers_openerless_preamble_until_close():
    state = NarrativeState(clip_duration_s=4.0, thinking_mode=True)
    state.apply_event(
        {
            "request_id": "r0",
            "kind": "completed",
            "text": "</think>\nPrevious narrative.",
        }
    )
    state.apply_event({"request_id": "r1", "kind": "started"})
    state.apply_event({"request_id": "r1", "kind": "delta", "text": "private analysis"})

    assert state.snapshot(0.0).narrative == "Previous narrative."

    state.apply_event({"request_id": "r1", "kind": "delta", "text": "</th" + "i"})
    assert state.snapshot(0.0).narrative == "Previous narrative."

    state.apply_event(
        {
            "request_id": "r1",
            "kind": "delta",
            "text": "nk>\nVisible narrative.",
        }
    )
    assert state.snapshot(0.0).narrative == "Visible narrative."

    state.apply_event({"request_id": "r2", "kind": "started"})
    state.apply_event({"request_id": "r2", "kind": "delta", "text": "unclosed analysis"})
    state.apply_event(
        {
            "request_id": "r2",
            "kind": "completed",
            "text": "unclosed analysis",
        }
    )
    assert state.snapshot(0.0).narrative == "Previous narrative."


def test_state_reports_model_error_and_ignores_unknown_events():
    state = NarrativeState(clip_duration_s=4.0)

    assert not state.apply_event({"request_id": "r1", "kind": "future_event"})
    assert state.apply_event({"request_id": "r1", "kind": "error", "message": "timeout"})
    snapshot = state.snapshot(0.0)
    assert snapshot.model_label == "MODEL · ERROR"
    assert snapshot.narrative == ""
    assert snapshot.working_message == "Reasoning unavailable: timeout"


def test_wrap_narrative_is_bounded_and_preserves_words():
    text = "one two three four five six seven eight nine ten eleven twelve"

    lines = wrap_narrative(text, width=15, max_lines=2)

    assert lines == ("one two three", "four five six…")
    assert all(len(line) <= 15 for line in lines)


def test_wrap_narrative_can_preserve_the_latest_streamed_text():
    text = "one two three four five six seven eight nine ten eleven twelve"

    lines = wrap_narrative(text, width=15, max_lines=2, keep_tail=True)

    assert lines == ("…ten eleven", "twelve")
    assert all(len(line) <= 15 for line in lines)


@pytest.mark.parametrize(
    ("max_lines", "keep_tail", "expected"),
    [
        (2, False, ("abcde", "fghij")),
        (1, False, ("abcd\u2026",)),
        (1, True, ("\u2026ghij",)),
    ],
)
def test_wrap_narrative_breaks_oversized_tokens(max_lines, keep_tail, expected):
    lines = wrap_narrative(
        "abcdefghij",
        width=5,
        max_lines=max_lines,
        keep_tail=keep_tail,
    )

    assert lines == expected
    assert all(len(line) <= 5 for line in lines)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 1}, "width"),
        ({"max_lines": 0}, "max_lines"),
    ],
)
def test_wrap_narrative_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        wrap_narrative("text", **kwargs)
