# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

"""State and text formatting for the procedural narrator display."""

from __future__ import annotations

import re
import textwrap
import threading
from dataclasses import dataclass
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)
_THINK_WITHOUT_OPEN = re.compile(r"\A.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"


def _strip_thinking_blocks(text: str) -> str:
    """Remove model thinking blocks, including one still arriving over SSE."""
    visible = _THINK_BLOCK.sub("", text)
    visible = _THINK_WITHOUT_OPEN.sub("", visible)
    visible = _UNCLOSED_THINK_BLOCK.sub("", visible)

    # Do not briefly render a fragmented opening tag between streamed deltas.
    lower_visible = visible.lower()
    for length in range(len(_THINK_OPEN_TAG) - 1, 0, -1):
        if lower_visible.endswith(_THINK_OPEN_TAG[:length]):
            visible = visible[:-length]
            break
    return visible.lstrip()


def _visible_narrative(text: str, *, thinking_mode: bool) -> str:
    """Return displayable text, buffering an explicitly enabled thinking preamble."""
    if thinking_mode and _THINK_CLOSE_TAG not in text.lower():
        return ""
    return _strip_thinking_blocks(text)


@dataclass(frozen=True)
class NarrativeSnapshot:
    """Display-ready state for one UI update."""

    input_live: bool
    input_label: str
    model_phase: str
    model_label: str
    narrative: str
    working_message: str = ""


class NarrativeState:
    """Track video availability and one in-flight reasoning response."""

    def __init__(
        self,
        clip_duration_s: float,
        input_timeout_s: float = 1.0,
        max_frame_gap_s: float | None = None,
        thinking_mode: bool = False,
    ):
        if clip_duration_s <= 0:
            raise ValueError("clip_duration_s must be positive")
        if input_timeout_s <= 0:
            raise ValueError("input_timeout_s must be positive")
        if max_frame_gap_s is not None and max_frame_gap_s <= 0:
            raise ValueError("max_frame_gap_s must be positive")

        self.clip_duration_s = clip_duration_s
        self.input_timeout_s = input_timeout_s
        self.max_frame_gap_s = max_frame_gap_s
        self.thinking_mode = thinking_mode
        self.first_frame_at: float | None = None
        self.last_frame_at: float | None = None
        self.active_request_id: str | None = None
        self.model_phase = "idle"
        self.stream_text = ""
        self._raw_stream_text = ""
        self.completed_text = ""
        self.error_message = ""
        self._lock = threading.Lock()

    def note_frame(self, now: float):
        """Update timestamps used to detect whether the video input is live."""
        with self._lock:
            collection_restarted = (
                self.max_frame_gap_s is not None
                and self.last_frame_at is not None
                and now - self.last_frame_at > self.max_frame_gap_s
            )
            if self.first_frame_at is None or collection_restarted:
                self.first_frame_at = now
            self.last_frame_at = now

    def apply_event(self, event: dict[str, Any]) -> bool:
        """Apply one reasoner event, returning whether it was recognised."""
        with self._lock:
            kind = event.get("kind")
            request_id = str(event.get("request_id", "unknown"))

            if kind == "started":
                self.active_request_id = request_id
                self.model_phase = "waiting"
                self.stream_text = ""
                self._raw_stream_text = ""
                self.error_message = ""
            elif kind == "delta":
                if request_id != self.active_request_id:
                    self.active_request_id = request_id
                    self.stream_text = ""
                    self._raw_stream_text = ""
                self.model_phase = "narrating"
                self._raw_stream_text += str(event.get("text", ""))
                self.stream_text = _visible_narrative(
                    self._raw_stream_text,
                    thinking_mode=self.thinking_mode,
                )
                self.error_message = ""
            elif kind == "completed":
                text = _visible_narrative(
                    str(event.get("text", "")),
                    thinking_mode=self.thinking_mode,
                ).strip()
                if text:
                    self.completed_text = text
                self.stream_text = ""
                self._raw_stream_text = ""
                self.active_request_id = None
                self.model_phase = "ready"
                self.error_message = ""
            elif kind == "error":
                self.active_request_id = None
                self.model_phase = "error"
                self.stream_text = ""
                self._raw_stream_text = ""
                self.error_message = str(event.get("message", "Unknown reasoning error"))
            else:
                return False
            return True

    def snapshot(self, now: float) -> NarrativeSnapshot:
        """Return the status and narrative that should currently be displayed."""
        with self._lock:
            input_live = (
                self.last_frame_at is not None and now - self.last_frame_at <= self.input_timeout_s
            )
            input_label = "VIDEO · LIVE" if input_live else "VIDEO · NO SIGNAL"

            phase = self.model_phase
            if phase == "idle":
                phase, model_label = self._idle_phase(now)
            else:
                model_label = {
                    "waiting": "MODEL · WAITING",
                    "narrating": "MODEL · NARRATING",
                    "ready": "MODEL · READY",
                    "error": "MODEL · ERROR",
                }[phase]

            if phase == "narrating" and self.stream_text:
                narrative = self.stream_text
            else:
                narrative = self.completed_text

            if phase == "error":
                working_message = f"Reasoning unavailable: {self.error_message}"
            elif phase == "waiting":
                working_message = "Temporal observation submitted. Waiting for the model."
            elif phase == "narrating":
                working_message = "Receiving the current procedural narrative."
            elif phase == "collecting":
                working_message = "Collecting temporal context for the first procedural narrative."
            elif not input_live:
                working_message = "Waiting for a replay or camera feed."
            elif self.completed_text:
                working_message = "Collecting temporal context for the next procedural narrative."
            else:
                working_message = "Preparing the first temporal observation."

            return NarrativeSnapshot(
                input_live=input_live,
                input_label=input_label,
                model_phase=phase,
                model_label=model_label,
                narrative=narrative,
                working_message=working_message,
            )

    def _idle_phase(self, now: float) -> tuple[str, str]:
        if self.first_frame_at is None:
            return "idle", "MODEL · IDLE"
        progress = min(max((now - self.first_frame_at) / self.clip_duration_s, 0.0), 1.0)
        if progress < 1.0:
            return "collecting", f"MODEL · COLLECTING {round(progress * 100):d}%"
        return "ready", "MODEL · READY"


def wrap_narrative(
    text: str,
    width: int = 72,
    max_lines: int = 3,
    *,
    keep_tail: bool = False,
) -> tuple[str, ...]:
    """Wrap display text and truncate it to a fixed-size narrative card."""
    if width <= 1:
        raise ValueError("width must be greater than one")
    if max_lines <= 0:
        raise ValueError("max_lines must be positive")

    normalised = " ".join(text.split())
    if not normalised:
        return ("",)

    lines = textwrap.wrap(
        normalised,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if len(lines) <= max_lines:
        return tuple(lines)

    if keep_tail:
        visible = lines[-max_lines:]
        words = visible[0].split()
        while len(words) > 1 and len(f"…{' '.join(words)}") > width:
            words.pop(0)
        first = " ".join(words)
        if len(first) >= width:
            first = first[-(width - 1) :].lstrip()
        visible[0] = f"…{first}"
        return tuple(visible)

    visible = lines[:max_lines]
    last = visible[-1]
    if len(last) >= width:
        last = last[: width - 1].rstrip()
    visible[-1] = f"{last}…"
    return tuple(visible)
