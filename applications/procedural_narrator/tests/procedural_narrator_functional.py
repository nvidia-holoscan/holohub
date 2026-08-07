# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

"""Run the complete procedural narrator graph against a local SSE endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import yaml

import applications.procedural_narrator.procedural_narrator as narrator_module
from applications.procedural_narrator.narrative_state import NarrativeState

NARRATION = "The endoscope advances while an instrument moves toward the tissue."


class ReasoningHandler(BaseHTTPRequestHandler):
    """Validate one temporal request and return a small streamed narration."""

    protocol_version = "HTTP/1.0"

    def do_POST(self):
        try:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            content = payload["messages"][0]["content"]
            video_items = [item for item in content if item.get("type") == "video_url"]
            if len(video_items) != 1:
                raise ValueError("request must contain exactly one video_url item")
            data_url = video_items[0]["video_url"]["url"]
            prefix, encoded = data_url.split(",", 1)
            if prefix != "data:video/mp4;base64":
                raise ValueError("request did not contain an MP4 data URL")
            video = base64.b64decode(encoded, validate=True)
            if len(video) < 12 or video[4:8] != b"ftyp":
                raise ValueError("request did not contain a valid MP4 header")
            if payload.get("stream") is not True:
                raise ValueError("request did not enable streaming")
            if payload.get("chat_template_kwargs") != {"enable_thinking": False}:
                raise ValueError("request did not disable thinking mode")
            self.server.request_payload = payload
        except Exception as error:  # noqa: BLE001 - report any fixture validation failure.
            self.server.request_error = str(error)
            self.send_error(400, str(error))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for text in ("The endoscope advances while ", "an instrument moves toward the tissue."):
            event = {"choices": [{"delta": {"content": text}}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format_string, *args):
        """Suppress HTTP access logs in CTest output."""


class TrackingNarrativeState(NarrativeState):
    """Expose graph progress without replacing any application operator."""

    event_kinds: ClassVar[list[str]] = []
    rendered: ClassVar[threading.Event] = threading.Event()

    def apply_event(self, event):
        accepted = super().apply_event(event)
        if accepted:
            self.event_kinds.append(str(event.get("kind")))
        return accepted

    def snapshot(self, now):
        snapshot = super().snapshot(now)
        if snapshot.narrative == NARRATION and snapshot.model_phase == "ready":
            self.rendered.set()
        return snapshot


def make_config(path: Path):
    """Write a fast, finite configuration for the downloaded replay."""
    config = {
        "replayer_source": {
            "basename": "surgical_video",
            "frame_rate": 30,
            "repeat": False,
            "realtime": True,
            "count": 30,
        },
        "v4l2_source": {
            "device": "/dev/video0",
            "pixel_format": "YUYV",
        },
        "format_converter": {
            "out_tensor_name": "frame",
            "out_dtype": "rgb888",
            "resize_width": 160,
            "resize_height": 90,
        },
        "reasoner": {
            "endpoint": "http://127.0.0.1:1/v1/chat/completions",
            "model": "mock-temporal-reasoner",
            "prompt": "Describe visible change across this temporal clip.",
            "mode": "video",
            "tensor_name": "frame",
            "sample_fps": 10,
            "clip_duration_s": 0.3,
            "request_interval_s": 60,
            "max_tokens": 32,
            "max_response_chars": 4096,
            "connect_timeout_s": 3,
            "timeout_s": 10,
            "api_key_env": "REASONER_API_KEY",
            "stream": True,
            "request_options": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
            "ffmpeg": "ffmpeg",
        },
        "holoviz": {
            "width": 320,
            "height": 180,
            "window_title": "procedural_narrator_test",
            "fullscreen": False,
            "use_exclusive_display": False,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing the surgical_video GXF replay.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for suffix in ("gxf_entities", "gxf_index"):
        replay_file = args.data / f"surgical_video.{suffix}"
        if not replay_file.is_file():
            print(f"FAIL: replay fixture is missing: {replay_file}")
            return 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), ReasoningHandler)
    server.request_payload = None
    server.request_error = None
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    TrackingNarrativeState.event_kinds = []
    TrackingNarrativeState.rendered.clear()
    narrator_module.NarrativeState = TrackingNarrativeState
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

    run_finished = threading.Event()
    stop_failure: list[str] = []
    with tempfile.TemporaryDirectory(prefix="procedural-narrator-test-") as directory:
        config_path = Path(directory) / "procedural_narrator_test.yaml"
        make_config(config_path)
        app = narrator_module.ProceduralNarratorApp(
            str(args.data),
            source="replayer",
            headless=True,
            endpoint=endpoint,
        )
        app.config(str(config_path))

        def stop_after_render():
            deadline = time.monotonic() + 30
            while not run_finished.is_set() and time.monotonic() < deadline:
                if TrackingNarrativeState.rendered.wait(0.05):
                    app.stop_execution()
                    return
            if not run_finished.is_set():
                stop_failure.append(
                    "timed out waiting for the completed narration to reach HoloViz"
                )
                app.stop_execution()

        controller = threading.Thread(target=stop_after_render, daemon=True)
        controller.start()
        run_error = None
        try:
            app.run()
        except Exception as error:  # noqa: BLE001 - surface any application failure.
            run_error = error
        finally:
            run_finished.set()
            controller.join(timeout=2)
            server.shutdown()
            server_thread.join(timeout=2)
            server.server_close()

    if run_error is not None:
        print(f"FAIL: application raised {run_error}")
        return 1
    if stop_failure:
        print(
            f"FAIL: {stop_failure[0]}; "
            f"request_received={server.request_payload is not None}; "
            f"request_error={server.request_error!r}; "
            f"events={TrackingNarrativeState.event_kinds}"
        )
        return 1
    if server.request_error is not None:
        print(f"FAIL: mock endpoint rejected request: {server.request_error}")
        return 1
    if server.request_payload is None:
        print("FAIL: mock endpoint did not receive a request")
        return 1
    if TrackingNarrativeState.event_kinds != ["started", "delta", "delta", "completed"]:
        print(f"FAIL: unexpected event sequence: {TrackingNarrativeState.event_kinds}")
        return 1

    print("SUCCESS: replay, temporal reasoner, SSE narration, and headless HoloViz completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
