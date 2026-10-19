# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

"""Online image and video reasoning operator."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import queue
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import requests
from holoscan.core import ConditionType, Operator, OperatorSpec
from requests.adapters import HTTPAdapter

from .media import encode_jpeg, encode_mp4, validate_rgb_frame

LOGGER = logging.getLogger(__name__)
_RESERVED_REQUEST_OPTIONS = {"max_tokens", "messages", "model", "stream"}
_RESPONSE_ENVELOPE_ALLOWANCE_BYTES = 64 * 1024
_MAX_HEADER_HELPER_THREADS = 2
_RESPONSE_READ_CHUNK_SIZE_BYTES = 8 * 1024


def _validate_positive_finite(name: str, value: float) -> None:
    """Require a positive finite timing value."""
    try:
        valid = not isinstance(value, bool) and math.isfinite(value) and value > 0
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a positive finite number")


def _sse_read_chunk_size(response: requests.Response) -> int:
    """Avoid buffering sparse close-delimited streams while batching chunked SSE."""
    raw_response = getattr(response, "raw", None)
    if getattr(raw_response, "chunked", False):
        return _RESPONSE_READ_CHUNK_SIZE_BYTES
    headers = getattr(response, "headers", {})
    transfer_encoding = headers.get("Transfer-Encoding", "")
    if any(value.strip().lower() == "chunked" for value in transfer_encoding.split(",")):
        return _RESPONSE_READ_CHUNK_SIZE_BYTES
    # urllib3 waits for the requested amount on close-delimited bodies. A byte
    # read preserves immediate delivery for sparse legacy SSE responses.
    return 1


def _close_response_handle(handle: tuple[requests.Response, requests.Session]) -> None:
    """Close a streamed response and its owning session."""
    response, session = handle
    try:
        response.close()
    finally:
        session.close()


def _shutdown_and_close(network_socket: socket.socket | None) -> None:
    """Interrupt a socket duplicate and release its descriptor."""
    if network_socket is None:
        return
    try:
        network_socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    finally:
        network_socket.close()


class _SocketInterrupter:
    """Keep a cancellable duplicate of the socket used to acquire headers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._cancelled = False

    def attach(self, network_socket: socket.socket) -> None:
        """Capture a socket, immediately interrupting it after cancellation."""
        try:
            duplicate = network_socket.dup()
        except OSError as error:
            network_socket.close()
            raise RuntimeError("could not prepare reasoning request cancellation") from error

        with self._lock:
            previous = self._socket
            if self._cancelled:
                interrupt = duplicate
            else:
                self._socket = duplicate
                interrupt = None
        if previous is not None:
            previous.close()
        _shutdown_and_close(interrupt)

    def cancel(self) -> None:
        """Interrupt the captured socket and any socket attached later."""
        with self._lock:
            self._cancelled = True
            network_socket = self._socket
            self._socket = None
        _shutdown_and_close(network_socket)

    def close(self) -> None:
        """Release the duplicate without affecting a successful response."""
        with self._lock:
            network_socket = self._socket
            self._socket = None
        if network_socket is not None:
            network_socket.close()


def _capturing_pool_class(pool_class: type, interrupter: _SocketInterrupter) -> type:
    """Wrap one urllib3 pool class so its newly connected socket is captured."""
    connection_class = pool_class.ConnectionCls

    class _CapturingConnection(connection_class):
        def _new_conn(self):
            network_socket = super()._new_conn()
            interrupter.attach(network_socket)
            return network_socket

    class _CapturingPool(pool_class):
        ConnectionCls = _CapturingConnection

    return _CapturingPool


class _InterruptibleHTTPAdapter(HTTPAdapter):
    """Install per-session urllib3 pools that expose pre-header sockets."""

    def __init__(self, interrupter: _SocketInterrupter) -> None:
        self._interrupter = interrupter
        super().__init__()

    def _configure_manager(self, manager: Any) -> None:
        if getattr(manager, "online_reasoner_interrupter", None) is self._interrupter:
            return
        manager.pool_classes_by_scheme = {
            scheme: _capturing_pool_class(pool_class, self._interrupter)
            for scheme, pool_class in manager.pool_classes_by_scheme.items()
        }
        manager.online_reasoner_interrupter = self._interrupter

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)
        self._configure_manager(self.poolmanager)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        self._configure_manager(manager)
        return manager


def _validate_endpoint(endpoint: str) -> bool:
    """Validate the endpoint and report whether it uses local cleartext HTTP."""
    if not isinstance(endpoint, str):
        raise TypeError("endpoint must be a valid HTTP or HTTPS URL")
    try:
        parsed = urlsplit(endpoint)
        hostname = parsed.hostname
        # Accessing port validates both its syntax and range.
        _ = parsed.port
    except ValueError as error:
        raise ValueError("endpoint must be a valid HTTP or HTTPS URL") from error

    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise ValueError("endpoint must be a valid HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None or "\\" in parsed.netloc:
        raise ValueError("endpoint must not include user information")
    if parsed.scheme == "https":
        return False

    try:
        address = ip_address(hostname)
    except ValueError:
        address = None
    if address is None or not address.is_loopback:
        raise ValueError(
            "HTTP endpoints must use a literal loopback address; "
            "use HTTPS for hostnames and non-local endpoints"
        )
    return True


def build_chat_payload(
    *,
    model: str,
    prompt: str,
    media_type: str,
    media: bytes,
    max_tokens: int,
    stream: bool,
    request_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one OpenAI-compatible multimodal chat-completion request."""
    if media_type not in {"image", "video"}:
        raise ValueError("media_type must be 'image' or 'video'")

    options = dict(request_options or {})
    reserved = _RESERVED_REQUEST_OPTIONS.intersection(options)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(f"request_options cannot override: {names}")

    mime_type = "image/jpeg" if media_type == "image" else "video/mp4"
    item_type = "image_url" if media_type == "image" else "video_url"
    data_url = f"data:{mime_type};base64,{base64.b64encode(media).decode('ascii')}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": item_type, item_type: {"url": data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    payload.update(options)
    return payload


def extract_completion_text(payload: dict[str, Any]) -> str:
    """Extract text from a non-streaming chat-completion response."""
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response does not contain choices[0].message.content") from error
    if not isinstance(text, str) or not text:
        raise ValueError("response content must be a non-empty string")
    return text


def iter_sse_text(lines: Iterable[str | bytes]) -> Iterable[str]:
    """Yield OpenAI-compatible text deltas from an SSE response."""
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8")
        else:
            line = raw_line
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].lstrip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
            choices = event["choices"]
            if choices == [] and "usage" in event:
                continue
            content = choices[0]["delta"].get("content")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("invalid chat-completion SSE event") from error
        if content is not None:
            if not isinstance(content, str):
                raise ValueError("SSE delta content must be a string")
            if content:
                yield content
    raise ValueError("SSE stream ended before [DONE]")


class OnlineVideoReasonerOp(Operator):
    """Periodically submit an image or bounded MP4 window to a reasoning endpoint."""

    def __init__(
        self,
        *args,
        endpoint: str,
        model: str,
        prompt: str,
        mode: str = "video",
        tensor_name: str = "",
        sample_fps: float = 4.0,
        clip_duration_s: float = 4.0,
        request_interval_s: float = 4.0,
        max_frame_gap_s: float | None = None,
        max_tokens: int = 128,
        max_response_chars: int = 1_048_576,
        connect_timeout_s: float = 10.0,
        timeout_s: float = 60.0,
        api_key_env: str = "REASONER_API_KEY",
        stream: bool = True,
        request_options: dict[str, Any] | None = None,
        ffmpeg: str = "ffmpeg",
        **kwargs,
    ):
        if mode not in {"image", "video"}:
            raise ValueError("mode must be 'image' or 'video'")
        uses_local_http = _validate_endpoint(endpoint)
        for field_name, value in (("model", model), ("prompt", prompt)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        _validate_positive_finite("sample_fps", sample_fps)
        _validate_positive_finite("clip_duration_s", clip_duration_s)
        _validate_positive_finite("request_interval_s", request_interval_s)
        if max_frame_gap_s is not None:
            _validate_positive_finite("max_frame_gap_s", max_frame_gap_s)
            if mode == "video" and max_frame_gap_s < 1.0 / sample_fps:
                raise ValueError(
                    "max_frame_gap_s must be at least one sample period (1 / sample_fps)"
                )
        _validate_positive_finite("connect_timeout_s", connect_timeout_s)
        _validate_positive_finite("timeout_s", timeout_s)
        if not isinstance(stream, bool):
            raise TypeError("stream must be a Boolean")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if (
            not isinstance(max_response_chars, int)
            or isinstance(max_response_chars, bool)
            or max_response_chars <= 0
        ):
            raise ValueError("max_response_chars must be a positive integer")
        frame_count = round(sample_fps * clip_duration_s)
        if mode == "video" and frame_count < 2:
            raise ValueError("video mode requires at least two frames per clip")
        options = dict(request_options or {})
        reserved = _RESERVED_REQUEST_OPTIONS.intersection(options)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"request_options cannot override: {names}")

        self.endpoint = endpoint
        self.model = model
        self.prompt = prompt
        self.mode = mode
        self.tensor_name = tensor_name
        self.sample_fps = sample_fps
        self.clip_duration_s = clip_duration_s
        self.request_interval_s = request_interval_s
        self.max_frame_gap_s = max_frame_gap_s or 2.0 / sample_fps
        self.max_tokens = max_tokens
        self.max_response_chars = max_response_chars
        self.connect_timeout_s = connect_timeout_s
        self.timeout_s = timeout_s
        self.api_key_env = api_key_env
        self.stream = stream
        self.request_options = options
        self.ffmpeg = ffmpeg
        self._trust_environment = not uses_local_http
        self._frame_count = frame_count
        self._frames: deque[np.ndarray] = deque(maxlen=frame_count)
        self._frame_times: deque[float] = deque(maxlen=frame_count)
        self._events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
        self._events_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[None] | None = None
        self._stop_event = threading.Event()
        self._response_lock = threading.Lock()
        self._active_response: requests.Response | None = None
        self._post_threads_lock = threading.Lock()
        self._post_threads: set[threading.Thread] = set()
        self._last_sample_at = float("-inf")
        self._next_sample_at = float("-inf")
        self._last_request_at = float("-inf")
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # A PeriodicCondition drives this operator. The optional input lets it
        # continue draining SSE events after a finite video source stops.
        spec.input("input").condition(ConditionType.NONE)
        spec.output("events")

    def start(self):
        self._frames.clear()
        self._frame_times.clear()
        self._clear_events()
        self._stop_event.clear()
        self._future = None
        self._active_response = None
        self._last_sample_at = float("-inf")
        self._next_sample_at = float("-inf")
        self._last_request_at = float("-inf")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-reasoner")

    def stop(self):
        self._stop_event.set()
        with self._response_lock:
            response = self._active_response
        if response is not None:
            self._interrupt_response(response)
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._future = None

    def compute(self, op_input, op_output, context):
        self._finish_completed_future()
        message = op_input.receive("input")
        if message is not None:
            self._accept_frame(message, time.monotonic(), input_context=op_input)
        self._emit_available_events(op_output)

    def _finish_completed_future(self):
        if self._future is None or not self._future.done():
            return
        try:
            self._future.result()
        except Exception as error:  # noqa: BLE001 - surface unexpected worker failures.
            self._push_event(
                {
                    "request_id": "internal",
                    "kind": "error",
                    "sequence": 0,
                    "message": f"reasoning worker failed: {error}",
                }
            )
        self._future = None

    def _accept_frame(self, message: Any, now: float, input_context: Any | None = None):
        sample_period = 1.0 / self.sample_fps
        if now < self._next_sample_at:
            return
        if (
            self.mode == "video"
            and self._frames
            and now - self._last_sample_at > self.max_frame_gap_s
        ):
            self._clear_video_window()

        frame = self._to_host_rgb(message, input_context=input_context)
        self._last_sample_at = now
        if math.isinf(self._next_sample_at):
            self._next_sample_at = now + sample_period
        else:
            elapsed_periods = math.floor((now - self._next_sample_at) / sample_period) + 1
            self._next_sample_at += max(1, elapsed_periods) * sample_period

        if self.mode == "video":
            self._frames.append(frame)
            self._frame_times.append(now)
            media_ready = len(self._frames) == self._frame_count
            if media_ready and not self._video_window_has_valid_timing(sample_period):
                self._clear_video_window()
                self._frames.append(frame)
                self._frame_times.append(now)
                media_ready = False
            media: np.ndarray | list[np.ndarray] = list(self._frames)
        else:
            media_ready = True
            media = frame

        request_ready = now - self._last_request_at >= self.request_interval_s
        if media_ready and request_ready and self._future is None:
            if self._executor is None:
                raise RuntimeError("operator has not been started")
            request_id = uuid.uuid4().hex
            self._last_request_at = now
            self._future = self._executor.submit(self._run_request, request_id, media)

    def _clear_video_window(self):
        self._frames.clear()
        self._frame_times.clear()

    def _video_window_has_valid_timing(self, sample_period: float) -> bool:
        expected_span = (self._frame_count - 1) * sample_period
        actual_span = self._frame_times[-1] - self._frame_times[0]
        return abs(actual_span - expected_span) <= sample_period

    def _to_host_rgb(self, message: Any, input_context: Any | None = None) -> np.ndarray:
        try:
            tensor = message.get(self.tensor_name)
        except AttributeError:
            tensor = message[self.tensor_name]
        if tensor is None:
            raise ValueError(f"input does not contain tensor {self.tensor_name!r}")

        is_cuda = hasattr(tensor, "__cuda_array_interface__")
        if is_cuda:
            import cupy as cp

            if input_context is None:
                frame = cp.asnumpy(cp.asarray(tensor))
            else:
                stream = input_context.receive_cuda_stream("input", allocate=False)
                if stream is not None:
                    with cp.cuda.ExternalStream(stream):
                        frame = cp.asnumpy(cp.asarray(tensor))
                else:
                    frame = cp.asnumpy(cp.asarray(tensor))
        else:
            frame = np.asarray(tensor)
        validate_rgb_frame(frame)
        if is_cuda:
            return frame
        return np.array(frame, copy=True, order="C")

    def _run_request(self, request_id: str, media: np.ndarray | list[np.ndarray]):
        sequence = 0
        deltas_dropped = False
        try:
            if self.mode == "image":
                encoded = encode_jpeg(
                    media,
                    ffmpeg=self.ffmpeg,
                    cancel_event=self._stop_event,
                )
            else:
                encoded = encode_mp4(
                    media,
                    self.sample_fps,
                    ffmpeg=self.ffmpeg,
                    cancel_event=self._stop_event,
                )
            payload = build_chat_payload(
                model=self.model,
                prompt=self.prompt,
                media_type=self.mode,
                media=encoded,
                max_tokens=self.max_tokens,
                stream=self.stream,
                request_options=self.request_options,
            )
            headers = {"Content-Type": "application/json"}
            api_key = os.environ.get(self.api_key_env, "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            if self._stop_event.is_set():
                return

            self._push_event(
                {
                    "request_id": request_id,
                    "kind": "started",
                    "mode": self.mode,
                    "frame_count": 1 if self.mode == "image" else len(media),
                }
            )
            deadline = time.monotonic() + self.timeout_s
            response_handle = self._post_interruptibly(payload, headers, deadline=deadline)
            if response_handle is None:
                return
            response, _ = response_handle
            with self._response_lock:
                if self._stop_event.is_set():
                    _close_response_handle(response_handle)
                    return
                self._active_response = response
            try:
                response.raise_for_status()
                if self.stream:
                    completion = io.StringIO()
                    completion_chars = 0
                    lines = self._iter_sse_lines(response, deadline=deadline)
                    try:
                        for text in iter_sse_text(lines):
                            if self._stop_event.is_set():
                                return
                            next_chars = completion_chars + len(text)
                            if next_chars > self.max_response_chars:
                                raise ValueError(
                                    f"response exceeded max_response_chars ({self.max_response_chars})"
                                )
                            completion.write(text)
                            completion_chars = next_chars
                            queued = self._push_event(
                                {
                                    "request_id": request_id,
                                    "kind": "delta",
                                    "sequence": sequence,
                                    "text": text,
                                }
                            )
                            deltas_dropped = deltas_dropped or not queued
                            sequence += 1
                    finally:
                        lines.close()
                    completed_text = completion.getvalue()
                    if not completed_text:
                        raise ValueError("stream completed without text")
                else:
                    response_payload = self._read_non_streaming_json(response, deadline=deadline)
                    if response_payload is None:
                        return
                    completed_text = extract_completion_text(response_payload)
                    if len(completed_text) > self.max_response_chars:
                        raise ValueError(
                            f"response exceeded max_response_chars ({self.max_response_chars})"
                        )

            finally:
                with self._response_lock:
                    if self._active_response is response:
                        self._active_response = None
                _close_response_handle(response_handle)

            if self._stop_event.is_set():
                return
            self._push_event(
                {
                    "request_id": request_id,
                    "kind": "completed",
                    "sequence": sequence,
                    "text": completed_text,
                    "deltas_dropped": deltas_dropped,
                }
            )
        except Exception as error:  # noqa: BLE001 - report every request worker failure.
            if not self._stop_event.is_set():
                self._push_event(
                    {
                        "request_id": request_id,
                        "kind": "error",
                        "sequence": sequence,
                        "message": str(error),
                    }
                )

    def _iter_response_chunks(
        self,
        response: requests.Response,
        *,
        response_name: str,
        chunk_size: int = _RESPONSE_READ_CHUNK_SIZE_BYTES,
        deadline: float | None = None,
    ) -> Iterable[bytes]:
        """Yield raw response chunks within the configured total deadline."""
        if deadline is None:
            deadline = time.monotonic() + self.timeout_s
        deadline_expired = threading.Event()
        timeout_message = (
            f"{response_name} response exceeded timeout_s ({self.timeout_s:g} seconds)"
        )

        def interrupt_at_deadline():
            deadline_expired.set()
            self._interrupt_response(response)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            interrupt_at_deadline()
            raise TimeoutError(timeout_message)

        deadline_timer = threading.Timer(remaining, interrupt_at_deadline)
        deadline_timer.daemon = True
        deadline_timer.start()
        try:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if self._stop_event.is_set():
                    return
                if deadline_expired.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError(timeout_message)
                if chunk:
                    yield chunk
            if self._stop_event.is_set():
                return
            if deadline_expired.is_set() or time.monotonic() >= deadline:
                raise TimeoutError(timeout_message)
        except Exception as error:
            if deadline_expired.is_set():
                raise TimeoutError(timeout_message) from error
            raise
        finally:
            deadline_timer.cancel()

    def _iter_sse_lines(
        self,
        response: requests.Response,
        *,
        deadline: float | None = None,
    ) -> Iterable[bytes]:
        """Frame bounded SSE lines directly from raw response chunks."""
        line = bytearray()
        line_limit = self.max_response_chars + _RESPONSE_ENVELOPE_ALLOWANCE_BYTES
        chunks = self._iter_response_chunks(
            response,
            response_name="SSE",
            chunk_size=_sse_read_chunk_size(response),
            deadline=deadline,
        )
        try:
            for chunk in chunks:
                start = 0
                while start < len(chunk):
                    newline = chunk.find(b"\n", start)
                    end = len(chunk) if newline < 0 else newline
                    segment = chunk[start:end]
                    if len(line) + len(segment) > line_limit:
                        raise ValueError(f"SSE line exceeded limit ({line_limit} bytes)")
                    line.extend(segment)
                    if newline < 0:
                        break
                    yield bytes(line)
                    line.clear()
                    start = newline + 1
            if line and not self._stop_event.is_set():
                yield bytes(line)
        finally:
            chunks.close()

    def _read_non_streaming_json(
        self,
        response: requests.Response,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any] | None:
        """Read one JSON response within bounded memory and wall-clock time."""
        body = bytearray()
        body_limit = self.max_response_chars + _RESPONSE_ENVELOPE_ALLOWANCE_BYTES
        chunks = self._iter_response_chunks(
            response,
            response_name="non-streaming",
            deadline=deadline,
        )
        try:
            for chunk in chunks:
                if len(body) + len(chunk) > body_limit:
                    raise ValueError(
                        f"non-streaming response body exceeded limit ({body_limit} bytes)"
                    )
                body.extend(chunk)
        finally:
            chunks.close()

        if self._stop_event.is_set():
            return None
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("non-streaming response is not valid JSON") from error
        if not isinstance(payload, dict):
            raise TypeError("non-streaming response must be a JSON object")
        return payload

    def _post_interruptibly(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        deadline: float | None = None,
    ) -> tuple[requests.Response, requests.Session] | None:
        if deadline is None:
            deadline = time.monotonic() + self.timeout_s
        timeout_message = f"reasoning request exceeded timeout_s ({self.timeout_s:g} seconds)"
        results: queue.Queue[tuple[requests.Response, requests.Session] | Exception] = queue.Queue(
            maxsize=1
        )
        handoff_lock = threading.Lock()
        abandoned = False
        interrupter = _SocketInterrupter()
        session = requests.Session()
        session.trust_env = self._trust_environment
        session.mount("http://", _InterruptibleHTTPAdapter(interrupter))
        session.mount("https://", _InterruptibleHTTPAdapter(interrupter))

        def abandon_request():
            nonlocal abandoned
            with handoff_lock:
                abandoned = True
                try:
                    pending_result = results.get_nowait()
                except queue.Empty:
                    pending_result = None
            interrupter.cancel()
            if isinstance(pending_result, tuple):
                _close_response_handle(pending_result)
            else:
                session.close()

        def post_request():
            try:
                with handoff_lock:
                    skip_request = (
                        abandoned or self._stop_event.is_set() or time.monotonic() >= deadline
                    )
                if skip_request:
                    session.close()
                    return

                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        session.close()
                        return
                    response = session.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                        # Keep transport validation bound to the configured URL.
                        allow_redirects=False,
                        # Always defer reading the body so stop() can interrupt
                        # streaming and non-streaming API responses alike.
                        stream=True,
                        timeout=(min(self.connect_timeout_s, remaining), self.timeout_s),
                    )
                    result: tuple[requests.Response, requests.Session] | Exception = (
                        response,
                        session,
                    )
                except Exception as error:  # noqa: BLE001 - hand off every thread failure.
                    session.close()
                    result = error

                with handoff_lock:
                    if abandoned:
                        if isinstance(result, tuple):
                            _close_response_handle(result)
                    else:
                        results.put_nowait(result)
            finally:
                try:
                    interrupter.close()
                finally:
                    with self._post_threads_lock:
                        self._post_threads.discard(threading.current_thread())

        request_thread = threading.Thread(
            target=post_request,
            name="online-reasoner-http",
            daemon=True,
        )
        with self._post_threads_lock:
            if len(self._post_threads) >= _MAX_HEADER_HELPER_THREADS:
                session.close()
                raise RuntimeError("previous reasoning requests are still terminating")
            self._post_threads.add(request_thread)
            try:
                request_thread.start()
            except Exception:
                self._post_threads.remove(request_thread)
                session.close()
                raise

        while True:
            if self._stop_event.is_set():
                abandon_request()
                return None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                abandon_request()
                raise TimeoutError(timeout_message)
            try:
                result = results.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                if isinstance(result, tuple):
                    _close_response_handle(result)
                return None
            if time.monotonic() >= deadline:
                if isinstance(result, tuple):
                    _close_response_handle(result)
                raise TimeoutError(timeout_message)
            if isinstance(result, Exception):
                raise result
            return result

    @staticmethod
    def _interrupt_response(response: requests.Response):
        """Interrupt a blocking response read without waiting for its read timeout."""
        try:
            socket_fd = os.dup(response.raw.fileno())
            with socket.socket(fileno=socket_fd) as connection:
                connection.shutdown(socket.SHUT_RDWR)
        except (AttributeError, OSError, ValueError):
            LOGGER.debug("Could not interrupt active reasoning response", exc_info=True)

    def _push_event(self, event: dict[str, Any]) -> bool:
        with self._events_lock:
            try:
                self._events.put_nowait(event)
                return True
            except queue.Full:
                # Partial deltas may be dropped under backpressure because the
                # completed event always carries the full response.
                if event["kind"] == "delta":
                    return False

            dropped = self._drop_newest_queued_delta()
            if dropped is None:
                # A bounded queue cannot retain unlimited canonical events.
                # Preserve those already queued instead of evicting one.
                return False

            dropped_request_id = dropped.get("request_id")
            if (
                event["kind"] == "completed"
                and dropped_request_id is not None
                and event.get("request_id") == dropped_request_id
            ):
                event["deltas_dropped"] = True
            try:
                self._events.put_nowait(event)
            except queue.Full:
                return False
            return False

    def _drop_newest_queued_delta(self) -> dict[str, Any] | None:
        """Remove one queued delta while retaining canonical events and order."""
        queued = []
        while True:
            try:
                queued.append(self._events.get_nowait())
                self._events.task_done()
            except queue.Empty:
                break

        dropped_index = next(
            (
                index
                for index in range(len(queued) - 1, -1, -1)
                if queued[index].get("kind") == "delta"
            ),
            None,
        )
        dropped = queued.pop(dropped_index) if dropped_index is not None else None
        if dropped is not None:
            dropped_request_id = dropped.get("request_id")
            if dropped_request_id is not None:
                for queued_event in queued:
                    if (
                        queued_event.get("kind") == "completed"
                        and queued_event.get("request_id") == dropped_request_id
                    ):
                        queued_event["deltas_dropped"] = True

        for queued_event in queued:
            self._events.put_nowait(queued_event)
        return dropped

    def _clear_events(self) -> None:
        """Discard events retained from an earlier operator run."""
        with self._events_lock:
            while True:
                try:
                    self._events.get_nowait()
                    self._events.task_done()
                except queue.Empty:
                    return

    def _emit_available_events(self, op_output):
        # The default Holoscan output connector has capacity one. Emit one
        # event per periodic tick and leave the remainder in the bounded queue.
        with self._events_lock:
            try:
                event = self._events.get_nowait()
                self._events.task_done()
            except queue.Empty:
                return
        op_output.emit(event, "events")
