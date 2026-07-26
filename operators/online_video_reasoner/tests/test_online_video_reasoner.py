# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

"""Protocol, HTTP, and Holoscan graph tests for OnlineVideoReasonerOp."""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

pytest.importorskip("holoscan")

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec

from operators.online_video_reasoner.online_video_reasoner import (
    OnlineVideoReasonerOp,
    build_chat_payload,
    extract_completion_text,
    iter_sse_text,
)


class _ReasoningHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(payload)

        self.send_response(200)
        if payload["stream"]:
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for text in ("A vehicle ", "moves left."):
                event = {"choices": [{"delta": {"content": text}}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"choices": [{"message": {"content": "A static scene."}}]}
            self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass


class _StalledSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        event = {"choices": [{"delta": {"content": "partial"}}]}
        self.wfile.write(f"data: {json.dumps(event)}\n\n: keepalive\n\n".encode())
        self.wfile.flush()
        self.server.request_received.set()
        self.server.release_response.wait()

    def log_message(self, format, *args):
        pass


@pytest.fixture
def reasoning_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReasoningHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture
def stalled_sse_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StalledSSEHandler)
    server.request_received = threading.Event()
    server.release_response = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release_response.set()
        server.shutdown()
        thread.join()
        server.server_close()


def _operator(server, *, mode, stream):
    return OnlineVideoReasonerOp(
        Application(),
        endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        model="test-model",
        prompt="Describe change.",
        mode=mode,
        sample_fps=4,
        clip_duration_s=0.5,
        request_interval_s=1,
        stream=stream,
    )


def _queued_events(operator):
    events = []
    while True:
        try:
            events.append(operator._events.get_nowait())
        except queue.Empty:
            return events


def test_operator_initialises_ports_and_bounded_frame_ring(reasoning_server):
    operator = _operator(reasoning_server, mode="video", stream=True)

    assert operator.mode == "video"
    assert operator._frames.maxlen == 2
    assert set(operator.spec.inputs) == {"input"}
    assert set(operator.spec.outputs) == {"events"}


def test_frame_ring_owns_cpu_input_buffer(reasoning_server):
    operator = _operator(reasoning_server, mode="video", stream=True)
    source = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    expected = source.copy()

    operator._accept_frame({"": source}, now=0.0)
    stored = operator._frames[0]
    source.fill(0)

    assert stored.flags.c_contiguous
    assert not np.shares_memory(stored, source)
    np.testing.assert_array_equal(stored, expected)


def test_video_window_resets_after_source_gap(reasoning_server):
    operator = OnlineVideoReasonerOp(
        Application(),
        endpoint=f"http://127.0.0.1:{reasoning_server.server_port}/v1/chat/completions",
        model="test-model",
        prompt="Describe change.",
        mode="video",
        sample_fps=4,
        clip_duration_s=1,
        request_interval_s=1,
    )
    first = np.zeros((12, 16, 3), dtype=np.uint8)
    after_gap = np.full((12, 16, 3), 173, dtype=np.uint8)

    operator._accept_frame({"": first}, now=0.0)
    operator._accept_frame({"": after_gap}, now=0.75)

    assert list(operator._frame_times) == [0.75]
    assert len(operator._frames) == 1
    np.testing.assert_array_equal(operator._frames[0], after_gap)


def test_video_window_rejects_sustained_low_rate_samples(reasoning_server):
    operator = OnlineVideoReasonerOp(
        Application(),
        endpoint=f"http://127.0.0.1:{reasoning_server.server_port}/v1/chat/completions",
        model="test-model",
        prompt="Describe change.",
        mode="video",
        sample_fps=4,
        clip_duration_s=1,
        request_interval_s=1,
    )

    for value, timestamp in enumerate((0.0, 0.4, 0.8, 1.2)):
        frame = np.full((12, 16, 3), value, dtype=np.uint8)
        operator._accept_frame({"": frame}, now=timestamp)

    assert list(operator._frame_times) == [1.2]
    assert len(operator._frames) == 1
    np.testing.assert_array_equal(operator._frames[0], np.full((12, 16, 3), 3, dtype=np.uint8))


def test_operator_waits_for_upstream_cuda_stream():
    cp = pytest.importorskip("cupy")
    from holoscan.resources import CudaStreamPool

    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device is required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device is required")

    delayed_fill = cp.RawKernel(
        r"""
        extern "C" __global__
        void delayed_fill(
            unsigned char* frame,
            unsigned long long delay_cycles,
            unsigned char value,
            int size)
        {
            const unsigned long long start = clock64();
            while (clock64() - start < delay_cycles) {
            }
            for (int index = 0; index < size; ++index) {
                frame[index] = value;
            }
        }
        """,
        "delayed_fill",
    )

    class DefaultStreamView:
        """Expose device data without advertising its producing stream."""

        def __init__(self, array):
            self._array = array

        @property
        def __cuda_array_interface__(self):
            interface = dict(self._array.__cuda_array_interface__)
            interface["stream"] = 1
            return interface

    class DelayedFrameSource(Operator):
        def setup(self, spec: OperatorSpec):
            spec.output("output")

        def compute(self, op_input, op_output, context):
            frame = cp.zeros((12, 16, 3), dtype=cp.uint8)
            cp.cuda.get_current_stream().synchronize()
            stream = context.allocate_cuda_stream("delayed_frame_stream")
            with cp.cuda.ExternalStream(stream):
                delayed_fill(
                    (1,),
                    (1,),
                    (frame, np.uint64(100_000_000), np.uint8(173), np.int32(frame.size)),
                )
            op_output.set_cuda_stream(stream, "output")
            op_output.emit({"frame": DefaultStreamView(frame)}, "output")

    class EventSink(Operator):
        def setup(self, spec: OperatorSpec):
            spec.input("events")

        def compute(self, op_input, op_output, context):
            op_input.receive("events")

    class StreamSyncApp(Application):
        def compose(self):
            stream_pool = CudaStreamPool(
                self,
                name="stream_pool",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=2,
            )
            source = DelayedFrameSource(
                self,
                stream_pool,
                CountCondition(self, count=1),
                name="source",
            )
            self.reasoner = OnlineVideoReasonerOp(
                self,
                CountCondition(self, count=20),
                PeriodicCondition(self, recess_period=timedelta(milliseconds=1)),
                name="reasoner",
                endpoint="http://127.0.0.1:1/v1/chat/completions",
                model="test-model",
                prompt="Describe change.",
                mode="video",
                tensor_name="frame",
                sample_fps=1000,
                clip_duration_s=0.002,
                request_interval_s=1,
                stream=True,
            )
            event_sink = EventSink(self, name="event_sink")
            self.add_flow(source, self.reasoner, {("output", "input")})
            self.add_flow(self.reasoner, event_sink, {("events", "events")})

    app = StreamSyncApp()
    app.run()

    assert len(app.reasoner._frames) == 1
    np.testing.assert_array_equal(
        app.reasoner._frames[0], np.full((12, 16, 3), 173, dtype=np.uint8)
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mode": "frames"}, "mode"),
        ({"endpoint": "grpc://localhost"}, "HTTP or HTTPS"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"sample_fps": 1, "clip_duration_s": 1}, "at least two frames"),
        ({"max_frame_gap_s": 0}, "max_frame_gap_s"),
    ],
)
def test_operator_rejects_invalid_configuration(override, message):
    options = {
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "model": "test-model",
        "prompt": "Describe change.",
        "mode": "video",
        "sample_fps": 4,
        "clip_duration_s": 0.5,
        "request_interval_s": 1,
    }
    options.update(override)

    with pytest.raises(ValueError, match=message):
        OnlineVideoReasonerOp(Application(), **options)


@pytest.mark.parametrize(
    ("media_type", "content_type", "mime_type"),
    [
        ("image", "image_url", "image/jpeg"),
        ("video", "video_url", "video/mp4"),
    ],
)
def test_build_chat_payload_uses_one_multimodal_media_item(media_type, content_type, mime_type):
    payload = build_chat_payload(
        model="model",
        prompt="prompt",
        media_type=media_type,
        media=b"media",
        max_tokens=32,
        stream=True,
        request_options={"temperature": 0},
    )

    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "prompt"}
    assert content[1]["type"] == content_type
    data_url = content[1][content_type]["url"]
    prefix, encoded = data_url.split(",", 1)
    assert prefix == f"data:{mime_type};base64"
    assert base64.b64decode(encoded) == b"media"
    assert payload["temperature"] == 0


def test_request_options_cannot_replace_protocol_fields():
    with pytest.raises(ValueError, match="messages"):
        build_chat_payload(
            model="model",
            prompt="prompt",
            media_type="image",
            media=b"media",
            max_tokens=32,
            stream=True,
            request_options={"messages": []},
        )


def test_sse_parser_yields_text_deltas_and_stops_at_done():
    lines = [
        b": keepalive",
        b"",
        b'data: {"choices":[{"delta":{"content":"one "}}]}',
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        b'data: {"choices":[{"delta":{"content":"two"}}]}',
        b"data: [DONE]",
        b'data: {"choices":[{"delta":{"content":"ignored"}}]}',
    ]
    assert list(iter_sse_text(lines)) == ["one ", "two"]


def test_completion_parser_requires_text():
    assert (
        extract_completion_text({"choices": [{"message": {"content": "description"}}]})
        == "description"
    )
    with pytest.raises(ValueError, match="does not contain"):
        extract_completion_text({"choices": []})


def test_video_request_posts_one_mp4_and_emits_sse_events(reasoning_server):
    operator = _operator(reasoning_server, mode="video", stream=True)
    frames = [
        np.full((12, 16, 3), (255, 0, 0), dtype=np.uint8),
        np.full((12, 16, 3), (0, 255, 0), dtype=np.uint8),
    ]

    operator._run_request("video-request", frames)

    payload = reasoning_server.requests[0]
    content = payload["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "video_url"]
    data_url = content[1]["video_url"]["url"]
    assert data_url.startswith("data:video/mp4;base64,")
    encoded = base64.b64decode(data_url.split(",", 1)[1])
    assert encoded[4:8] == b"ftyp"
    assert _queued_events(operator) == [
        {
            "request_id": "video-request",
            "kind": "started",
            "mode": "video",
            "frame_count": 2,
        },
        {
            "request_id": "video-request",
            "kind": "delta",
            "sequence": 0,
            "text": "A vehicle ",
        },
        {
            "request_id": "video-request",
            "kind": "delta",
            "sequence": 1,
            "text": "moves left.",
        },
        {
            "request_id": "video-request",
            "kind": "completed",
            "sequence": 2,
            "text": "A vehicle moves left.",
            "deltas_dropped": False,
        },
    ]


def test_image_request_posts_jpeg_and_emits_completed_event(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)
    frame = np.zeros((12, 16, 3), dtype=np.uint8)

    operator._run_request("image-request", frame)

    payload = reasoning_server.requests[0]
    content = payload["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    data_url = content[1]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = base64.b64decode(data_url.split(",", 1)[1])
    assert encoded.startswith(b"\xff\xd8")
    assert _queued_events(operator) == [
        {
            "request_id": "image-request",
            "kind": "started",
            "mode": "image",
            "frame_count": 1,
        },
        {
            "request_id": "image-request",
            "kind": "completed",
            "sequence": 0,
            "text": "A static scene.",
            "deltas_dropped": False,
        },
    ]


def test_started_event_follows_media_encoding(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    encoding_started = threading.Event()
    release_encoding = threading.Event()

    def blocking_encoder(frame, *, ffmpeg):
        encoding_started.set()
        release_encoding.wait()
        return b"\xff\xd8\xff\xd9"

    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        blocking_encoder,
    )
    worker = threading.Thread(
        target=operator._run_request,
        args=("delayed-encoding", np.zeros((12, 16, 3), dtype=np.uint8)),
    )
    worker.start()
    assert encoding_started.wait(timeout=1)
    assert _queued_events(operator) == []

    release_encoding.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert [event["kind"] for event in _queued_events(operator)] == ["started", "completed"]


def test_stop_interrupts_stalled_sse_request(stalled_sse_server):
    operator = _operator(stalled_sse_server, mode="image", stream=True)
    operator.start()
    operator._accept_frame({"": np.zeros((12, 16, 3), dtype=np.uint8)}, now=0.0)
    assert stalled_sse_server.request_received.wait(timeout=2)
    deadline = time.monotonic() + 1
    while operator._events.qsize() < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert operator._events.qsize() == 2

    stopped = threading.Event()

    def stop_operator():
        operator.stop()
        stopped.set()

    started_at = time.monotonic()
    stop_thread = threading.Thread(target=stop_operator)
    stop_thread.start()
    try:
        assert stopped.wait(timeout=1)
        assert time.monotonic() - started_at < 1
    finally:
        stalled_sse_server.release_response.set()
        stop_thread.join(timeout=2)

    assert not stop_thread.is_alive()
    assert [event["kind"] for event in _queued_events(operator)] == ["started", "delta"]


def test_stop_interrupts_request_before_response_headers(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=True)
    request_started = threading.Event()
    release_request = threading.Event()
    request_finished = threading.Event()

    def pending_request(*args, **kwargs):
        request_started.set()
        try:
            release_request.wait()
            raise RuntimeError("request released")
        finally:
            request_finished.set()

    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.requests.post",
        pending_request,
    )
    operator.start()
    operator._accept_frame({"": np.zeros((12, 16, 3), dtype=np.uint8)}, now=0.0)
    assert request_started.wait(timeout=1)

    started_at = time.monotonic()
    try:
        operator.stop()
        assert time.monotonic() - started_at < 1
    finally:
        release_request.set()

    assert request_finished.wait(timeout=1)
    assert [event["kind"] for event in _queued_events(operator)] == ["started"]


def test_compute_drains_events_without_an_input_message(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator._events.put({"request_id": "r", "kind": "completed", "sequence": 0, "text": "done"})
    operator._events.put({"request_id": "s", "kind": "completed", "sequence": 0, "text": "later"})

    class EmptyInput:
        def receive(self, port):
            assert port == "input"
            return None

    class CapturingOutput:
        def __init__(self):
            self.events = []

        def emit(self, event, port):
            assert port == "events"
            self.events.append(event)

    output = CapturingOutput()
    operator.compute(EmptyInput(), output, None)
    assert output.events == [
        {"request_id": "r", "kind": "completed", "sequence": 0, "text": "done"}
    ]
    assert operator._events.qsize() == 1


def test_completed_event_marks_dropped_stream_deltas(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator._events = queue.Queue(maxsize=1)
    assert operator._push_event({"kind": "delta", "text": "partial"})

    completed = {"kind": "completed", "text": "complete", "deltas_dropped": False}
    assert not operator._push_event(completed)

    assert _queued_events(operator) == [
        {"kind": "completed", "text": "complete", "deltas_dropped": True}
    ]


def test_completed_event_tolerates_concurrent_queue_refill(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)

    class ConcurrentlyRefilledQueue:
        def put_nowait(self, event):
            raise queue.Full

        def get_nowait(self):
            return {"kind": "started"}

    operator._events = ConcurrentlyRefilledQueue()
    completed = {"kind": "completed", "text": "complete", "deltas_dropped": False}

    assert not operator._push_event(completed)
    assert completed["deltas_dropped"]


def test_holoscan_graph_streams_video_reasoning_events(reasoning_server):
    class FrameSource(Operator):
        def __init__(self, *args, **kwargs):
            self.index = 0
            super().__init__(*args, **kwargs)

        def setup(self, spec: OperatorSpec):
            spec.output("output")

        def compute(self, op_input, op_output, context):
            frame = np.zeros((12, 16, 3), dtype=np.uint8)
            frame[:, :, self.index % 3] = 255
            self.index += 1
            op_output.emit({"frame": frame}, "output")

    class EventSink(Operator):
        def __init__(self, *args, **kwargs):
            self.events = []
            super().__init__(*args, **kwargs)

        def setup(self, spec: OperatorSpec):
            spec.input("events")

        def compute(self, op_input, op_output, context):
            self.events.append(op_input.receive("events"))

    class ReasoningApp(Application):
        def compose(self):
            source = FrameSource(
                self,
                CountCondition(self, count=2),
                PeriodicCondition(self, recess_period=timedelta(milliseconds=50)),
                name="source",
            )
            reasoner = OnlineVideoReasonerOp(
                self,
                CountCondition(self, count=200),
                PeriodicCondition(self, recess_period=timedelta(milliseconds=5)),
                name="reasoner",
                endpoint=(
                    f"http://127.0.0.1:{reasoning_server.server_port}" "/v1/chat/completions"
                ),
                model="test-model",
                prompt="Describe change.",
                mode="video",
                tensor_name="frame",
                sample_fps=25,
                clip_duration_s=0.08,
                request_interval_s=1,
                stream=True,
            )
            self.sink = EventSink(self, name="sink")
            self.add_flow(source, reasoner, {("output", "input")})
            self.add_flow(reasoner, self.sink, {("events", "events")})

    app = ReasoningApp()
    app.run()

    assert [event["kind"] for event in app.sink.events] == [
        "started",
        "delta",
        "delta",
        "completed",
    ]
    assert app.sink.events[-1]["text"] == "A vehicle moves left."
    assert len(reasoning_server.requests) == 1
