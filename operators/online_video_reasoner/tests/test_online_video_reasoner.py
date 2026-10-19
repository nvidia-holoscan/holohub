# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

"""Protocol, HTTP, and Holoscan graph tests for OnlineVideoReasonerOp."""

from __future__ import annotations

import base64
import json
import queue
import socket
import sys
import threading
import time
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

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
            usage = {"choices": [], "usage": {"completion_tokens": 4}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"choices": [{"message": {"content": "A static scene."}}]}
            self.wfile.write(json.dumps(response).encode())

    def log_message(self, format_string, *args):
        pass


class _StalledSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        event = {"choices": [{"delta": {"content": "partial"}}]}
        payload = f"data: {json.dumps(event)}\n\n: keepalive\n\n".encode()
        self.wfile.write(f"{len(payload):X}\r\n".encode())
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()
        self.server.request_received.set()
        self.server.release_response.wait()
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass

    def log_message(self, format_string, *args):
        pass


class _CloseDelimitedSSEHandler(BaseHTTPRequestHandler):
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

    def log_message(self, format_string, *args):
        pass


class _DripHeaderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        with self.server.request_lock:
            self.server.request_count += 1
            request_number = self.server.request_count

        if request_number > 1:
            response = json.dumps(
                {"choices": [{"message": {"content": "Recovered response."}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            return

        self.server.request_received.set()
        self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Drip: ")
        self.wfile.flush()
        while not self.server.release_response.wait(timeout=0.025):
            try:
                self.wfile.write(b"x")
                self.wfile.flush()
            except OSError:
                self.server.connection_closed.set()
                return
        try:
            self.wfile.write(b"\r\nContent-Length: 2\r\n\r\n{}")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format_string, *args):
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


@pytest.fixture
def close_delimited_sse_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CloseDelimitedSSEHandler)
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


@pytest.fixture
def drip_header_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DripHeaderHandler)
    server.request_received = threading.Event()
    server.release_response = threading.Event()
    server.connection_closed = threading.Event()
    server.request_lock = threading.Lock()
    server.request_count = 0
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


def _route_test_server_through_dns(operator, server):
    """Force the local HTTP fixture through DNS after endpoint validation."""
    operator.endpoint = f"http://reasoner.test:{server.server_port}/v1/chat/completions"


def _queued_events(operator):
    events = []
    while True:
        try:
            events.append(operator._events.get_nowait())
        except queue.Empty:
            return events


def _post_helper_count(operator):
    with operator._post_threads_lock:
        return len(operator._post_threads)


def _wait_for_post_helpers(operator, expected, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _post_helper_count(operator) == expected:
            return True
        time.sleep(0.01)
    return _post_helper_count(operator) == expected


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


def test_cuda_input_without_stream_uses_default_copy(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    expected = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    calls = []

    tensor = SimpleNamespace(__cuda_array_interface__={})

    def asarray(value):
        calls.append(("asarray", value))
        return expected

    def asnumpy(value):
        calls.append(("asnumpy", value))
        return value

    def external_stream(stream):
        pytest.fail(f"ExternalStream must not be constructed for {stream!r}")

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asarray=asarray,
            asnumpy=asnumpy,
            cuda=SimpleNamespace(ExternalStream=external_stream),
        ),
    )

    class InputContext:
        def receive_cuda_stream(self, port, *, allocate):
            assert port == "input"
            assert not allocate

    frame = operator._to_host_rgb({"": tensor}, input_context=InputContext())

    assert calls == [("asarray", tensor), ("asnumpy", expected)]
    assert frame is expected
    assert frame.flags.c_contiguous
    np.testing.assert_array_equal(frame, expected)


def test_cuda_default_stream_is_synchronised(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    expected = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    calls = []

    tensor = SimpleNamespace(__cuda_array_interface__={})

    def asarray(value):
        calls.append(("asarray", value))
        return expected

    def asnumpy(value):
        calls.append(("asnumpy", value))
        return value

    class StreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            calls.append(("enter", self.stream))

        def __exit__(self, *args):
            calls.append(("exit", self.stream))

    def external_stream(stream):
        calls.append(("external_stream", stream))
        return StreamContext(stream)

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asarray=asarray,
            asnumpy=asnumpy,
            cuda=SimpleNamespace(ExternalStream=external_stream),
        ),
    )

    class InputContext:
        def receive_cuda_stream(self, port, *, allocate):
            assert port == "input"
            assert not allocate
            return 0

    frame = operator._to_host_rgb({"": tensor}, input_context=InputContext())

    assert calls == [
        ("external_stream", 0),
        ("enter", 0),
        ("asarray", tensor),
        ("asnumpy", expected),
        ("exit", 0),
    ]
    assert frame is expected
    np.testing.assert_array_equal(frame, expected)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://reasoner.example/v1/chat/completions",
        "http://127.42.0.1:8000/v1/chat/completions",
        "http://[::1]:8000/v1/chat/completions",
    ],
)
def test_operator_accepts_https_and_explicit_loopback_http(endpoint):
    operator = OnlineVideoReasonerOp(
        Application(),
        endpoint=endpoint,
        model="test-model",
        prompt="Describe change.",
        mode="image",
    )

    assert operator.endpoint == endpoint


@pytest.mark.parametrize(
    ("endpoint", "trust_environment"),
    [
        ("http://127.0.0.1:8000/v1/chat/completions", False),
        ("https://reasoner.example/v1/chat/completions", True),
    ],
)
def test_transport_isolates_local_http_and_splits_timeouts(
    endpoint,
    trust_environment,
    monkeypatch,
):
    operator = OnlineVideoReasonerOp(
        Application(),
        endpoint=endpoint,
        model="test-model",
        prompt="Describe change.",
        mode="image",
        connect_timeout_s=3,
        timeout_s=17,
    )
    observed = {}

    class Response:
        def close(self):
            pass

    def post(session, url, **kwargs):
        observed["trust_env"] = session.trust_env
        observed["url"] = url
        observed["timeout"] = kwargs["timeout"]
        return Response()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.requests.Session.post",
        post,
    )

    response_handle = operator._post_interruptibly({}, {})

    assert response_handle is not None
    response, session = response_handle
    try:
        assert observed == {
            "trust_env": trust_environment,
            "url": endpoint,
            "timeout": (3, 17),
        }
    finally:
        response.close()
        session.close()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mode": "frames"}, "mode"),
        ({"endpoint": "grpc://localhost"}, "HTTP or HTTPS"),
        ({"endpoint": "http://reasoner.example/v1/chat/completions"}, "use HTTPS"),
        ({"endpoint": "http://192.0.2.1/v1/chat/completions"}, "use HTTPS"),
        ({"endpoint": "http://localhost:8000/v1/chat/completions"}, "use HTTPS"),
        ({"endpoint": "http://LOCALHOST:8000/v1/chat/completions"}, "use HTTPS"),
        ({"endpoint": "http://localhost.evil/v1/chat/completions"}, "use HTTPS"),
        ({"endpoint": "http://localhost./v1/chat/completions"}, "use HTTPS"),
        (
            {"endpoint": r"http://reasoner.example\@localhost/v1/chat/completions"},
            "user information",
        ),
        ({"endpoint": "https://:invalid/v1/chat/completions"}, "valid HTTP or HTTPS"),
        ({"max_tokens": 0}, "positive integer"),
        ({"max_tokens": 1.5}, "positive integer"),
        ({"max_tokens": True}, "positive integer"),
        ({"max_tokens": float("nan")}, "positive integer"),
        ({"max_response_chars": 0}, "max_response_chars"),
        ({"request_options": {"max_tokens": 64}}, "max_tokens"),
        ({"connect_timeout_s": 0}, "connect_timeout_s"),
        ({"sample_fps": 1, "clip_duration_s": 1}, "at least two frames"),
        ({"max_frame_gap_s": 0}, "max_frame_gap_s"),
        ({"max_frame_gap_s": 0.2}, "at least one sample period"),
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


@pytest.mark.parametrize("field", ["model", "prompt"])
@pytest.mark.parametrize("value", ["", None, 123, {"text": "Describe change."}])
def test_operator_rejects_invalid_required_strings(field, value):
    options = {
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "model": "test-model",
        "prompt": "Describe change.",
        "mode": "image",
    }
    options[field] = value

    with pytest.raises(ValueError, match=rf"^{field} must be a non-empty string$"):
        OnlineVideoReasonerOp(Application(), **options)


def test_operator_rejects_non_boolean_stream():
    with pytest.raises(TypeError, match="stream must be a Boolean"):
        OnlineVideoReasonerOp(
            Application(),
            endpoint="http://127.0.0.1:1/v1/chat/completions",
            model="test-model",
            prompt="Describe change.",
            mode="image",
            stream="false",
        )


@pytest.mark.parametrize(
    "field",
    [
        "sample_fps",
        "clip_duration_s",
        "request_interval_s",
        "max_frame_gap_s",
        "connect_timeout_s",
        "timeout_s",
    ],
)
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_operator_rejects_invalid_timing_configuration(field, value):
    options = {
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "model": "test-model",
        "prompt": "Describe change.",
        "mode": "video",
        "sample_fps": 4,
        "clip_duration_s": 0.5,
        "request_interval_s": 1,
    }
    options[field] = value

    with pytest.raises(ValueError, match=field):
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", 64),
        ("messages", []),
    ],
)
def test_request_options_cannot_replace_protocol_fields(field, value):
    with pytest.raises(ValueError, match=field):
        build_chat_payload(
            model="model",
            prompt="prompt",
            media_type="image",
            media=b"media",
            max_tokens=32,
            stream=True,
            request_options={field: value},
        )


def test_sse_parser_yields_text_deltas_and_stops_at_done():
    lines = [
        b": keepalive",
        b"",
        b'data: {"choices":[{"delta":{"content":"one "}}]}',
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        b'data: {"choices":[{"delta":{"content":"two"}}]}',
        b'data: {"choices":[],"usage":{"completion_tokens":2}}',
        b"data: [DONE]",
        b'data: {"choices":[{"delta":{"content":"ignored"}}]}',
    ]
    assert list(iter_sse_text(lines)) == ["one ", "two"]


def test_sse_parser_rejects_stream_without_done_marker():
    lines = [b'data: {"choices":[{"delta":{"content":"partial"}}]}']

    with pytest.raises(ValueError, match=r"before \[DONE\]"):
        list(iter_sse_text(lines))


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

    def blocking_encoder(frame, *, ffmpeg, cancel_event):
        assert not cancel_event.is_set()
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


@pytest.mark.parametrize(
    ("mode", "encoder_name"),
    [
        ("image", "encode_jpeg"),
        ("video", "encode_mp4"),
    ],
)
def test_stop_interrupts_media_encoding(reasoning_server, monkeypatch, mode, encoder_name):
    operator = _operator(reasoning_server, mode=mode, stream=False)
    encoding_started = threading.Event()

    def blocking_encoder(*args, ffmpeg, cancel_event):
        encoding_started.set()
        assert cancel_event.wait(timeout=2)
        return b"encoded"

    monkeypatch.setattr(
        f"operators.online_video_reasoner.online_video_reasoner.{encoder_name}",
        blocking_encoder,
    )
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    media = frame if mode == "image" else [frame, frame]
    operator.start()
    assert operator._executor is not None
    operator._future = operator._executor.submit(
        operator._run_request,
        "cancelled-encoding",
        media,
    )
    try:
        assert encoding_started.wait(timeout=1)
        started_at = time.monotonic()
        operator.stop()
        assert time.monotonic() - started_at < 1
    finally:
        if operator._executor is not None:
            operator.stop()

    assert _queued_events(operator) == []


def test_truncated_sse_emits_error_not_completed(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=True)

    class TruncatedResponse:
        def __init__(self):
            self.headers = {"Transfer-Encoding": "chunked"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}'

        def close(self):
            pass

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(
        operator,
        "_post_interruptibly",
        lambda payload, headers, *, deadline: (TruncatedResponse(), Session()),
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        lambda frame, *, ffmpeg, cancel_event: b"\xff\xd8\xff\xd9",
    )

    operator._run_request("truncated-stream", np.zeros((12, 16, 3), dtype=np.uint8))

    events = _queued_events(operator)
    assert [event["kind"] for event in events] == ["started", "delta", "error"]
    assert events[-1]["message"] == "SSE stream ended before [DONE]"


@pytest.mark.parametrize(
    ("parts", "limit", "expected_kinds"),
    [
        (("four",), 4, ["started", "delta", "completed"]),
        (("four", "!"), 4, ["started", "delta", "error"]),
    ],
)
def test_stream_response_respects_character_limit(
    reasoning_server,
    monkeypatch,
    parts,
    limit,
    expected_kinds,
):
    operator = _operator(reasoning_server, mode="image", stream=True)
    operator.max_response_chars = limit

    class Response:
        def __init__(self):
            self.headers = {"Transfer-Encoding": "chunked"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            for text in parts:
                event = {"choices": [{"delta": {"content": text}}]}
                yield f"data: {json.dumps(event)}\n".encode()
            yield b"data: [DONE]\n"

        def close(self):
            pass

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(
        operator,
        "_post_interruptibly",
        lambda payload, headers, *, deadline: (Response(), Session()),
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        lambda frame, *, ffmpeg, cancel_event: b"\xff\xd8\xff\xd9",
    )

    operator._run_request("bounded-stream", np.zeros((12, 16, 3), dtype=np.uint8))

    events = _queued_events(operator)
    assert [event["kind"] for event in events] == expected_kinds
    if expected_kinds[-1] == "completed":
        assert events[-1]["text"] == "four"
    else:
        assert events[-1]["message"] == "response exceeded max_response_chars (4)"


def test_sse_deadline_applies_to_heartbeat_lines(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=True)
    operator.timeout_s = 1
    moments = iter((0.0, 0.5, 1.0))

    class Response:
        def __init__(self):
            self.headers = {"Transfer-Encoding": "chunked"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            yield b": heartbeat\n"
            yield b": heartbeat\n"

        def close(self):
            pass

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(
        operator,
        "_post_interruptibly",
        lambda payload, headers, *, deadline: (Response(), Session()),
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        lambda frame, *, ffmpeg, cancel_event: b"\xff\xd8\xff\xd9",
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.time.monotonic",
        lambda: next(moments),
    )

    operator._run_request("timed-stream", np.zeros((12, 16, 3), dtype=np.uint8))

    events = _queued_events(operator)
    assert [event["kind"] for event in events] == ["started", "error"]
    assert events[-1]["message"] == "SSE response exceeded timeout_s (1 seconds)"


def test_sse_line_framing_handles_arbitrary_block_boundaries(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=True)
    content = "\u00e9"
    event = (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": content}}]},
            ensure_ascii=False,
        )
    ).encode()
    payload = b": heartbeat\r\n\r\n" + event + b"\r\n\r\ndata: [DONE]\n\n"
    split = payload.index(b"\xc3\xa9") + 1
    chunks = (payload[:split], payload[split : split + 11], payload[split + 11 :])

    class Response:
        def __init__(self):
            self.headers = {"Transfer-Encoding": "chunked"}

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            yield from chunks

    lines = operator._iter_sse_lines(Response())

    assert list(iter_sse_text(lines)) == [content]


def test_sse_unterminated_line_is_bounded_before_framing(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=True)
    operator.max_response_chars = 4

    class Response:
        def __init__(self):
            self.headers = {"Transfer-Encoding": "chunked"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            payload = b"x" * (4 + 64 * 1024 + 1)
            for start in range(0, len(payload), chunk_size):
                yield payload[start : start + chunk_size]

        def close(self):
            pass

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(
        operator,
        "_post_interruptibly",
        lambda payload, headers, *, deadline: (Response(), Session()),
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        lambda frame, *, ffmpeg, cancel_event: b"\xff\xd8\xff\xd9",
    )

    operator._run_request("bounded-sse-line", np.zeros((12, 16, 3), dtype=np.uint8))

    events = _queued_events(operator)
    assert [event["kind"] for event in events] == ["started", "error"]
    assert events[-1]["message"] == "SSE line exceeded limit (65540 bytes)"


def test_non_stream_response_respects_character_limit(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator.max_response_chars = 4

    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            payload = {"choices": [{"message": {"content": "five!"}}]}
            yield json.dumps(payload).encode()

        def close(self):
            pass

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(
        operator,
        "_post_interruptibly",
        lambda payload, headers, *, deadline: (Response(), Session()),
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        lambda frame, *, ffmpeg, cancel_event: b"\xff\xd8\xff\xd9",
    )

    operator._run_request("bounded-response", np.zeros((12, 16, 3), dtype=np.uint8))

    events = _queued_events(operator)
    assert [event["kind"] for event in events] == ["started", "error"]
    assert events[-1]["message"] == "response exceeded max_response_chars (4)"


def test_non_stream_response_body_is_bounded_before_parsing(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator.max_response_chars = 4

    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            yield b"x" * (4 + 64 * 1024 + 1)

        def close(self):
            pass

    class Session:
        def close(self):
            pass

    monkeypatch.setattr(
        operator,
        "_post_interruptibly",
        lambda payload, headers, *, deadline: (Response(), Session()),
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        lambda frame, *, ffmpeg, cancel_event: b"\xff\xd8\xff\xd9",
    )

    operator._run_request("bounded-body", np.zeros((12, 16, 3), dtype=np.uint8))

    events = _queued_events(operator)
    assert [event["kind"] for event in events] == ["started", "error"]
    assert events[-1]["message"] == ("non-streaming response body exceeded limit (65540 bytes)")


def test_header_wait_consumes_non_stream_response_deadline(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator.timeout_s = 1
    moments = iter((0.0, 0.75, 1.0))
    timer_delays = []

    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 8192
            yield b"{"
            yield b"}"

        def close(self):
            pass

    class Session:
        def close(self):
            pass

    class Timer:
        daemon = False

        def __init__(self, delay, callback):
            timer_delays.append(delay)

        def start(self):
            pass

        def cancel(self):
            pass

    def response_after_header_wait(payload, headers, *, deadline):
        assert deadline == 1.0
        return Response(), Session()

    monkeypatch.setattr(
        operator,
        "_post_interruptibly",
        response_after_header_wait,
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.encode_jpeg",
        lambda frame, *, ffmpeg, cancel_event: b"\xff\xd8\xff\xd9",
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.time.monotonic",
        lambda: next(moments),
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.threading.Timer",
        Timer,
    )

    operator._run_request("timed-response", np.zeros((12, 16, 3), dtype=np.uint8))

    events = _queued_events(operator)
    assert timer_delays == [0.25]
    assert [event["kind"] for event in events] == ["started", "error"]
    assert events[-1]["message"] == ("non-streaming response exceeded timeout_s (1 seconds)")


def test_stop_interrupts_stalled_sse_request(stalled_sse_server):
    operator = _operator(stalled_sse_server, mode="image", stream=True)
    operator.start()
    operator._accept_frame({"": np.zeros((12, 16, 3), dtype=np.uint8)}, now=0.0)
    assert stalled_sse_server.request_received.wait(timeout=2)
    events = [operator._events.get(timeout=1) for _ in range(2)]

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
    assert [event["kind"] for event in events] == ["started", "delta"]
    assert _queued_events(operator) == []


def test_close_delimited_sse_delivers_partial_event_before_eof(
    close_delimited_sse_server,
):
    operator = _operator(close_delimited_sse_server, mode="image", stream=True)
    operator.start()
    try:
        operator._accept_frame({"": np.zeros((12, 16, 3), dtype=np.uint8)}, now=0.0)
        assert close_delimited_sse_server.request_received.wait(timeout=2)
        events = [operator._events.get(timeout=1) for _ in range(2)]
        assert [event["kind"] for event in events] == ["started", "delta"]
        assert events[-1]["text"] == "partial"
    finally:
        close_delimited_sse_server.release_response.set()
        operator.stop()


def test_stop_interrupts_request_before_response_headers(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=True)
    request_started = threading.Event()
    release_request = threading.Event()
    request_finished = threading.Event()
    response_closed = threading.Event()
    session_closed = threading.Event()

    def pending_request(session, *args, **kwargs):
        assert kwargs["allow_redirects"] is False
        request_started.set()
        try:
            release_request.wait()
            return SimpleNamespace(close=response_closed.set)
        finally:
            request_finished.set()

    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.requests.Session.post",
        pending_request,
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.requests.Session.close",
        lambda session: session_closed.set(),
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
    assert response_closed.wait(timeout=1)
    assert session_closed.wait(timeout=1)
    assert [event["kind"] for event in _queued_events(operator)] == ["started"]


def test_header_deadline_bounds_drip_response(drip_header_server):
    operator = _operator(drip_header_server, mode="image", stream=False)
    operator.timeout_s = 0.2
    operator.start()
    started_at = time.monotonic()
    try:
        operator._accept_frame({"": np.zeros((12, 16, 3), dtype=np.uint8)}, now=0.0)
        assert drip_header_server.request_received.wait(timeout=1)
        assert operator._future is not None
        while not operator._future.done() and time.monotonic() - started_at < 1:
            time.sleep(0.01)
        assert operator._future.done()
        assert time.monotonic() - started_at < 1
        operator._finish_completed_future()

        events = _queued_events(operator)
        assert [event["kind"] for event in events] == ["started", "error"]
        assert events[-1]["message"] == ("reasoning request exceeded timeout_s (0.2 seconds)")
        assert drip_header_server.connection_closed.wait(timeout=1)

        assert _wait_for_post_helpers(operator, 0)

        operator._run_request("recovered", np.zeros((12, 16, 3), dtype=np.uint8))
        recovery_events = _queued_events(operator)
        assert [event["kind"] for event in recovery_events] == ["started", "completed"]
        assert recovery_events[-1]["text"] == "Recovered response."
        assert drip_header_server.request_count == 2
    finally:
        drip_header_server.release_response.set()
        operator.stop()


def test_header_deadline_closes_eventual_late_response(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator.timeout_s = 0.05
    request_started = threading.Event()
    release_request = threading.Event()
    response_closed = threading.Event()
    session_closed = threading.Event()

    def pending_request(session, *args, **kwargs):
        request_started.set()
        release_request.wait()
        return SimpleNamespace(close=response_closed.set)

    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.requests.Session.post",
        pending_request,
    )
    monkeypatch.setattr(
        "operators.online_video_reasoner.online_video_reasoner.requests.Session.close",
        lambda session: session_closed.set(),
    )
    operator.start()
    operator._accept_frame({"": np.zeros((12, 16, 3), dtype=np.uint8)}, now=0.0)
    assert request_started.wait(timeout=1)
    assert operator._future is not None
    try:
        assert operator._future.done() or operator._future.exception(timeout=1) is None
        operator._finish_completed_future()
        events = _queued_events(operator)
        assert [event["kind"] for event in events] == ["started", "error"]
        assert session_closed.wait(timeout=1)
        assert _post_helper_count(operator) == 1
    finally:
        release_request.set()

    assert response_closed.wait(timeout=1)
    assert _wait_for_post_helpers(operator, 0)
    operator.stop()


def test_dns_timeout_preserves_one_recovery_lane(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    _route_test_server_through_dns(operator, reasoning_server)
    operator.timeout_s = 0.2
    first_lookup_started = threading.Event()
    release_first_lookup = threading.Event()
    lookup_lock = threading.Lock()
    lookup_count = 0
    original_getaddrinfo = socket.getaddrinfo

    def controlled_getaddrinfo(host, port, *args, **kwargs):
        nonlocal lookup_count
        if host != "reasoner.test":
            return original_getaddrinfo(host, port, *args, **kwargs)
        with lookup_lock:
            lookup_count += 1
            lookup_number = lookup_count
        if lookup_number == 1:
            first_lookup_started.set()
            release_first_lookup.wait()
        return original_getaddrinfo("127.0.0.1", port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", controlled_getaddrinfo)
    payload = {"stream": False}
    headers = {"Content-Type": "application/json"}
    try:
        with pytest.raises(TimeoutError, match="reasoning request exceeded"):
            operator._post_interruptibly(payload, headers)
        assert first_lookup_started.is_set()
        assert _post_helper_count(operator) == 1

        response, session = operator._post_interruptibly(
            payload,
            headers,
            deadline=time.monotonic() + 1,
        )
        try:
            assert response.status_code == 200
        finally:
            response.close()
            session.close()
        assert lookup_count == 2
    finally:
        release_first_lookup.set()

    assert _wait_for_post_helpers(operator, 0)


def test_dns_helper_threads_remain_bounded(reasoning_server, monkeypatch):
    operator = _operator(reasoning_server, mode="image", stream=False)
    _route_test_server_through_dns(operator, reasoning_server)
    operator.timeout_s = 0.1
    lookup_started = (threading.Event(), threading.Event())
    release_lookups = threading.Event()
    lookup_lock = threading.Lock()
    lookup_count = 0
    original_getaddrinfo = socket.getaddrinfo

    def blocked_getaddrinfo(host, port, *args, **kwargs):
        nonlocal lookup_count
        if host != "reasoner.test":
            return original_getaddrinfo(host, port, *args, **kwargs)
        with lookup_lock:
            lookup_count += 1
            lookup_number = lookup_count
        lookup_started[lookup_number - 1].set()
        release_lookups.wait()
        return original_getaddrinfo("127.0.0.1", port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", blocked_getaddrinfo)
    payload = {"stream": False}
    headers = {"Content-Type": "application/json"}
    try:
        for started in lookup_started:
            with pytest.raises(TimeoutError, match="reasoning request exceeded"):
                operator._post_interruptibly(payload, headers)
            assert started.is_set()

        assert _post_helper_count(operator) == 2
        with pytest.raises(RuntimeError, match="previous reasoning requests are still terminating"):
            operator._post_interruptibly(payload, headers)
        assert lookup_count == 2
    finally:
        release_lookups.set()

    assert _wait_for_post_helpers(operator, 0)


def test_compute_drains_events_without_an_input_message(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator._events.put({"request_id": "r", "kind": "completed", "sequence": 0, "text": "done"})
    operator._events.put({"request_id": "s", "kind": "completed", "sequence": 0, "text": "later"})

    class EmptyInput:
        def receive(self, port):
            assert port == "input"

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


def test_start_discards_events_from_previous_run(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator.start()
    operator._events.put({"request_id": "stale", "kind": "completed", "text": "old"})
    operator.stop()

    operator.start()
    try:
        assert _queued_events(operator) == []
    finally:
        operator.stop()


def test_completed_event_marks_dropped_stream_deltas(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator._events = queue.Queue(maxsize=1)
    assert operator._push_event({"request_id": "r", "kind": "delta", "text": "partial"})

    completed = {
        "request_id": "r",
        "kind": "completed",
        "text": "complete",
        "deltas_dropped": False,
    }
    assert not operator._push_event(completed)

    assert _queued_events(operator) == [
        {
            "request_id": "r",
            "kind": "completed",
            "text": "complete",
            "deltas_dropped": True,
        }
    ]


def test_backpressure_preserves_queued_completion_and_marks_dropped_delta(
    reasoning_server,
):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator._events = queue.Queue(maxsize=2)
    operator._events.put({"request_id": "r", "kind": "delta", "text": "partial"})
    operator._events.put(
        {
            "request_id": "r",
            "kind": "completed",
            "text": "complete",
            "deltas_dropped": False,
        }
    )

    started = {"request_id": "next", "kind": "started"}
    assert not operator._push_event(started)

    assert _queued_events(operator) == [
        {
            "request_id": "r",
            "kind": "completed",
            "text": "complete",
            "deltas_dropped": True,
        },
        started,
    ]


def test_backpressure_does_not_evict_canonical_events(reasoning_server):
    operator = _operator(reasoning_server, mode="image", stream=False)
    operator._events = queue.Queue(maxsize=2)
    completed = {"request_id": "r", "kind": "completed", "text": "complete"}
    error = {"request_id": "s", "kind": "error", "message": "failed"}
    operator._events.put(completed)
    operator._events.put(error)

    assert not operator._push_event({"request_id": "next", "kind": "started"})
    assert _queued_events(operator) == [completed, error]


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
                endpoint=(f"http://127.0.0.1:{reasoning_server.server_port}/v1/chat/completions"),
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
