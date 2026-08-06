# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

"""Display ongoing clinical narration for a replay or V4L2 video source."""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import CudaStreamPool, UnboundedAllocator

from applications.procedural_narrator.narrative_state import (
    NarrativeSnapshot,
    NarrativeState,
    wrap_narrative,
)
from operators.online_video_reasoner import OnlineVideoReasonerOp

CARD_COLOR = [0.018, 0.027, 0.031, 0.88]
TEXT_COLOR = [1.0, 1.0, 1.0, 1.0]
DIM_TEXT_COLOR = [0.70, 0.76, 0.77, 1.0]
GREEN = [0.30, 0.90, 0.53, 1.0]
AMBER = [1.0, 0.72, 0.24, 1.0]
RED = [0.96, 0.30, 0.28, 1.0]
TRANSPARENT = [0.0, 0.0, 0.0, 0.0]
WORKING_PULSE_PERIOD_S = 3.0
WORKING_PULSE_MIN_ALPHA = 0.35
WORKING_PULSE_MAX_ALPHA = 0.75
REASONER_TICK_RATE_HZ = 60.0


def _validate_replayer_settings(replayer_args: dict[str, Any]) -> float:
    """Return an explicit replay cadence that is safe for temporal windows."""
    frame_rate = replayer_args.get("frame_rate", 0)
    try:
        valid_frame_rate = (
            not isinstance(frame_rate, bool) and math.isfinite(frame_rate) and frame_rate > 0
        )
    except TypeError:
        valid_frame_rate = False
    if not valid_frame_rate:
        raise ValueError("replayer_source.frame_rate must be a positive finite number")
    if replayer_args.get("realtime", True) is not True:
        raise ValueError(
            "replayer_source.realtime must be true so frame_rate defines playback cadence"
        )
    if replayer_args.get("repeat", False) is not False:
        raise ValueError(
            "replayer_source.repeat must be false to prevent reasoning windows "
            "from crossing replay boundaries"
        )
    return float(frame_rate)


def _filled_rectangle(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Return two normalised triangles forming a filled rectangle."""
    return np.asarray(
        [
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0, y0),
            (x1, y1),
            (x0, y1),
        ],
        dtype=np.float32,
    )


def _filled_circle(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    segments: int = 20,
) -> np.ndarray:
    """Return a triangle fan approximating a filled normalised circle."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments + 1, dtype=np.float32)
    coordinates = []
    for index in range(segments):
        coordinates.extend(
            (
                (center_x, center_y),
                (
                    center_x + radius_x * np.cos(angles[index]),
                    center_y + radius_y * np.sin(angles[index]),
                ),
                (
                    center_x + radius_x * np.cos(angles[index + 1]),
                    center_y + radius_y * np.sin(angles[index + 1]),
                ),
            )
        )
    return np.asarray(coordinates, dtype=np.float32)


def _add_geometry(
    tensors: dict[str, np.ndarray],
    specs: list[HolovizOp.InputSpec],
    *,
    name: str,
    coordinates: np.ndarray,
    geometry_type: str,
    color: list[float],
    priority: int,
    line_width: float = 1.0,
):
    tensors[name] = coordinates
    spec = HolovizOp.InputSpec(name, geometry_type)
    spec.color = color
    spec.priority = priority
    spec.line_width = line_width
    specs.append(spec)


def _add_text(
    tensors: dict[str, np.ndarray],
    specs: list[HolovizOp.InputSpec],
    *,
    name: str,
    lines: tuple[str, ...],
    x: float,
    y: float,
    size: float,
    line_height: float,
    color: list[float],
    priority: int = 30,
):
    tensors[name] = np.asarray(
        [(x, y + index * line_height, size) for index in range(len(lines))],
        dtype=np.float32,
    )
    spec = HolovizOp.InputSpec(name, "text")
    spec.text = list(lines)
    spec.color = color
    spec.priority = priority
    specs.append(spec)


def build_overlay(
    snapshot: NarrativeSnapshot,
    *,
    blink_on: bool,
    working_alpha: float = 1.0,
) -> tuple[dict[str, np.ndarray], list[HolovizOp.InputSpec]]:
    """Build one bottom narrative card with compact video and model indicators."""
    tensors: dict[str, np.ndarray] = {}
    specs: list[HolovizOp.InputSpec] = []

    _add_geometry(
        tensors,
        specs,
        name="narrative_card",
        coordinates=_filled_rectangle(0.15, 0.735, 0.85, 0.965),
        geometry_type="triangles",
        color=CARD_COLOR,
        priority=20,
    )
    _add_text(
        tensors,
        specs,
        name="narrative_heading",
        lines=("CLINICAL NARRATIVE",),
        x=0.18,
        y=0.770,
        size=0.027,
        line_height=0.0,
        color=GREEN,
    )
    _add_text(
        tensors,
        specs,
        name="narrative_text",
        lines=wrap_narrative(
            snapshot.narrative,
            keep_tail=snapshot.model_phase == "narrating",
        ),
        x=0.18,
        y=0.815,
        size=0.025,
        line_height=0.035,
        color=TEXT_COLOR,
    )

    model_fill = {
        "idle": TRANSPARENT,
        "collecting": TRANSPARENT,
        "waiting": AMBER if blink_on else TRANSPARENT,
        "narrating": GREEN,
        "ready": TRANSPARENT,
        "error": RED,
    }[snapshot.model_phase]
    for name, center_x, fill_color in (
        ("video_status", 0.795, GREEN if snapshot.input_live else TRANSPARENT),
        ("model_status", 0.820, model_fill),
    ):
        _add_geometry(
            tensors,
            specs,
            name=f"{name}_outline",
            coordinates=np.asarray([(center_x, 0.785, 0.014, 0.024)], dtype=np.float32),
            geometry_type="ovals",
            color=DIM_TEXT_COLOR,
            priority=28,
            line_width=2.0,
        )
        _add_geometry(
            tensors,
            specs,
            name=f"{name}_fill",
            coordinates=_filled_circle(center_x, 0.785, 0.0055, 0.0098),
            geometry_type="triangles",
            color=fill_color,
            priority=29,
        )

    _add_text(
        tensors,
        specs,
        name="working_text",
        lines=wrap_narrative(snapshot.working_message, width=88, max_lines=1),
        x=0.18,
        y=0.930,
        size=0.018,
        line_height=0.0,
        color=[*DIM_TEXT_COLOR[:3], working_alpha],
    )

    return tensors, specs


def working_pulse_alpha(now: float) -> float:
    """Return a slowly varying opacity for subordinate work-status text."""
    progress = (now % WORKING_PULSE_PERIOD_S) / WORKING_PULSE_PERIOD_S
    wave = (math.sin(2.0 * math.pi * progress) + 1.0) / 2.0
    return WORKING_PULSE_MIN_ALPHA + wave * (WORKING_PULSE_MAX_ALPHA - WORKING_PULSE_MIN_ALPHA)


class NarrativeEventSinkOp(Operator):
    """Consume reasoner events immediately into shared display state."""

    def __init__(self, fragment, state: NarrativeState, *args, **kwargs):
        self._state = state
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("events")

    def compute(self, op_input, op_output, context):
        event = op_input.receive("events")
        if event is not None:
            self._state.apply_event(event)


class NarrativeDisplayOp(Operator):
    """Cache the latest video frame and repaint video plus status periodically."""

    def __init__(
        self,
        fragment,
        *args,
        state: NarrativeState,
        frame_width: int,
        frame_height: int,
        input_tensor_name: str,
        **kwargs,
    ):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not isinstance(input_tensor_name, str):
            raise TypeError("input_tensor_name must be a string")
        self._state = state
        self._input_tensor_name = input_tensor_name
        self._latest_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("video").condition(ConditionType.NONE)
        spec.output("tensors")
        spec.output("specs")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("video")
        now = time.monotonic()
        if message is not None:
            tensor = message.get(self._input_tensor_name)
            if tensor is None:
                raise ValueError(f"video input does not contain tensor {self._input_tensor_name!r}")
            if hasattr(tensor, "__cuda_array_interface__"):
                import cupy as cp

                stream_ptr = op_input.receive_cuda_stream("video", allocate=False)
                if stream_ptr is not None:
                    with cp.cuda.ExternalStream(stream_ptr):
                        self._latest_frame = cp.asnumpy(cp.asarray(tensor))
                else:
                    self._latest_frame = cp.asnumpy(cp.asarray(tensor))
            else:
                self._latest_frame = np.array(tensor, copy=True, order="C")
            self._state.note_frame(now)

        tensors, specs = build_overlay(
            self._state.snapshot(now),
            blink_on=int(now * 2) % 2 == 0,
            working_alpha=working_pulse_alpha(now),
        )
        tensors["frame"] = self._latest_frame
        op_output.emit(tensors, "tensors")
        op_output.emit(specs, "specs")


class ProceduralNarratorApp(Application):
    """Narrate temporal change in a clinical replay or V4L2 camera feed."""

    def __init__(
        self,
        data_path: str,
        source: str = "replayer",
        video_device: str | None = None,
        headless: bool = False,
        endpoint: str | None = None,
    ):
        super().__init__()
        if source not in {"replayer", "v4l2"}:
            raise ValueError("source must be 'replayer' or 'v4l2'")
        self.name = "procedural_narrator"
        self._data_path = data_path
        self._source = source
        self._video_device = video_device
        self._headless = headless
        self._endpoint = endpoint

    def compose(self):
        allocator = UnboundedAllocator(self, name="allocator")
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        source, source_output, in_dtype, source_frame_rate = self._make_source(allocator)
        converter_args = dict(self.kwargs("format_converter"))
        converter_output_tensor_name = converter_args.get("out_tensor_name", "")
        if not isinstance(converter_output_tensor_name, str):
            raise TypeError("format_converter.out_tensor_name must be a string")
        converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=allocator,
            cuda_stream_pool=cuda_stream_pool,
            in_dtype=in_dtype,
            **converter_args,
        )

        reasoner_args = dict(self.kwargs("reasoner"))
        reasoner_args["tensor_name"] = converter_output_tensor_name
        if self._endpoint is not None:
            reasoner_args["endpoint"] = self._endpoint
        reasoner = OnlineVideoReasonerOp(
            self,
            PeriodicCondition(
                self,
                name="reasoner_periodic",
                recess_period=timedelta(seconds=1 / REASONER_TICK_RATE_HZ),
            ),
            name="reasoner",
            **reasoner_args,
        )
        if source_frame_rate is not None and reasoner.sample_fps > source_frame_rate:
            raise ValueError(
                "reasoner.sample_fps must not exceed replayer_source.frame_rate "
                f"({source_frame_rate:g} fps)"
            )
        if reasoner.sample_fps > REASONER_TICK_RATE_HZ:
            raise ValueError("reasoner.sample_fps must not exceed the 60 Hz reasoner tick rate")
        chat_template_kwargs = reasoner.request_options.get("chat_template_kwargs")
        thinking_mode = (
            chat_template_kwargs.get("enable_thinking", False)
            if isinstance(chat_template_kwargs, dict)
            else False
        )
        if not isinstance(thinking_mode, bool):
            raise TypeError("reasoner thinking mode must be true or false")
        state = NarrativeState(
            clip_duration_s=float(reasoner.clip_duration_s),
            max_frame_gap_s=float(reasoner.max_frame_gap_s),
            thinking_mode=thinking_mode,
        )
        event_sink = NarrativeEventSinkOp(self, state=state, name="narrative_event_sink")
        display = NarrativeDisplayOp(
            self,
            PeriodicCondition(
                self,
                name="display_periodic",
                recess_period=timedelta(seconds=1 / 60),
            ),
            state=state,
            frame_width=int(converter_args["resize_width"]),
            frame_height=int(converter_args["resize_height"]),
            input_tensor_name=converter_output_tensor_name,
            name="narrative_display",
        )

        holoviz_args = dict(self.kwargs("holoviz"))
        holoviz_args["headless"] = self._headless
        visualizer = HolovizOp(
            self,
            name="holoviz",
            allocator=allocator,
            cuda_stream_pool=cuda_stream_pool,
            tensors=[{"name": "frame", "type": "color", "priority": 0}],
            window_close_callback=self.on_window_closed,
            **holoviz_args,
        )

        self.add_flow(source, converter, {(source_output, "source_video")})
        self.add_flow(converter, reasoner, {("tensor", "input")})
        self.add_flow(converter, display, {("tensor", "video")})
        self.add_flow(reasoner, event_sink, {("events", "events")})
        self.add_flow(display, visualizer, {("tensors", "receivers")})
        self.add_flow(display, visualizer, {("specs", "input_specs")})

    def _make_source(
        self,
        allocator: UnboundedAllocator,
    ) -> tuple[Operator, str, str, float | None]:
        if self._source == "v4l2":
            source_args: dict[str, Any] = dict(self.kwargs("v4l2_source"))
            if self._video_device is not None:
                source_args["device"] = self._video_device
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=allocator,
                **source_args,
            )
            return source, "signal", "rgba8888", None

        source_args = dict(self.kwargs("replayer_source"))
        source_frame_rate = _validate_replayer_settings(source_args)
        source = VideoStreamReplayerOp(
            self,
            name="replayer_source",
            directory=self._data_path,
            **source_args,
        )
        return source, "output", "rgb888", source_frame_rate

    def on_window_closed(self):
        self.stop_execution()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Narrate a clinical replay or camera feed with temporal video reasoning."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("procedural_narrator.yaml")),
        help="Path to the application YAML configuration.",
    )
    parser.add_argument(
        "--source",
        choices=("replayer", "v4l2"),
        default="replayer",
        help="Use the downloaded replay or a V4L2-compatible camera.",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(os.environ.get("HOLOHUB_DATA_PATH", "../data"), "endoscopy"),
        help="Directory containing surgical_video.gxf_entities and surgical_video.gxf_index.",
    )
    parser.add_argument(
        "--video-device",
        default=None,
        help="Override the configured V4L2 device, for example /dev/video2.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run HoloViz without opening a display window.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override reasoner.endpoint from the YAML configuration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = ProceduralNarratorApp(
        args.data,
        source=args.source,
        video_device=args.video_device,
        headless=args.headless,
        endpoint=args.endpoint,
    )
    app.config(args.config)
    app.run()


if __name__ == "__main__":
    main()
