# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

"""Online image and video reasoning operator."""

from __future__ import annotations

import base64
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
from concurrent.futures import Future, ThreadPoolExecutor
from ipaddress import ip_address
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np
import requests
from holoscan.core import ConditionType, Operator, OperatorSpec

from .media import encode_jpeg, encode_mp4, validate_rgb_frame

LOGGER = logging.getLogger(__name__)
_RESERVED_REQUEST_OPTIONS = {"max_tokens", "messages", "model", "stream"}


def _close_response_handle(handle: tuple[requests.Response, requests.Session]) -> None:
    """Close a streamed response and its owning session."""
    response, session = handle
    try:
        response.close()
    finally:
        session.close()


def _validate_endpoint(endpoint: str) -> bool:
    """Validate the endpoint and report whether it uses local cleartext HTTP."""
    if not isinstance(endpoint, str):
        raise ValueError("endpoint must be a valid HTTP or HTTPS URL")
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

    if hostname == "localhost":
        return True
    try:
        address = ip_address(hostname)
    except ValueError:
        address = None
    if address is None or not address.is_loopback:
        raise ValueError(
            "HTTP endpoints must use localhost or a literal loopback address; "
            "use HTTPS for non-local endpoints"
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
        if not model or not prompt:
            raise ValueError("model and prompt are required")
        if sample_fps <= 0 or clip_duration_s <= 0 or request_interval_s <= 0:
            raise ValueError("sample_fps, clip_duration_s and request_interval_s must be positive")
        if max_frame_gap_s is not None and (
            not math.isfinite(max_frame_gap_s) or max_frame_gap_s <= 0
        ):
            raise ValueError("max_frame_gap_s must be a positive finite number")
        if max_tokens <= 0 or connect_timeout_s <= 0 or timeout_s <= 0:
            raise ValueError("max_tokens, connect_timeout_s and timeout_s must be positive")
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
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[None] | None = None
        self._stop_event = threading.Event()
        self._response_lock = threading.Lock()
        self._active_response: requests.Response | None = None
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
        except Exception as error:  # Defensive: _run_request normally handles errors.
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
                if stream:
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
                encoded = encode_jpeg(media, ffmpeg=self.ffmpeg)
            else:
                encoded = encode_mp4(media, self.sample_fps, ffmpeg=self.ffmpeg)
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
            response_handle = self._post_interruptibly(payload, headers)
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
                    chunks = []
                    # Larger buffers can delay events until EOF on non-chunked SSE responses.
                    lines = self._interruptible_lines(response.iter_lines(chunk_size=1))
                    for text in iter_sse_text(lines):
                        if self._stop_event.is_set():
                            return
                        chunks.append(text)
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
                    completed_text = "".join(chunks)
                    if not completed_text:
                        raise ValueError("stream completed without text")
                else:
                    completed_text = extract_completion_text(response.json())

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
        except Exception as error:
            if not self._stop_event.is_set():
                self._push_event(
                    {
                        "request_id": request_id,
                        "kind": "error",
                        "sequence": sequence,
                        "message": str(error),
                    }
                )

    def _post_interruptibly(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[requests.Response, requests.Session] | None:
        results: queue.Queue[tuple[requests.Response, requests.Session] | Exception] = queue.Queue(
            maxsize=1
        )
        handoff_lock = threading.Lock()
        abandoned = False

        def post_request():
            session: requests.Session | None = None
            try:
                session = requests.Session()
                session.trust_env = self._trust_environment
                response = session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    # Keep transport validation bound to the configured URL.
                    allow_redirects=False,
                    # Always defer reading the body so stop() can interrupt
                    # streaming and non-streaming API responses alike.
                    stream=True,
                    timeout=(self.connect_timeout_s, self.timeout_s),
                )
                result: tuple[requests.Response, requests.Session] | Exception = (
                    response,
                    session,
                )
            except Exception as error:
                if session is not None:
                    session.close()
                result = error

            with handoff_lock:
                if abandoned:
                    if isinstance(result, tuple):
                        _close_response_handle(result)
                else:
                    results.put_nowait(result)

        request_thread = threading.Thread(
            target=post_request,
            name="online-reasoner-http",
            daemon=True,
        )
        request_thread.start()

        while True:
            try:
                result = results.get(timeout=0.05)
            except queue.Empty:
                if not self._stop_event.is_set():
                    continue
                with handoff_lock:
                    abandoned = True
                    try:
                        pending_result = results.get_nowait()
                    except queue.Empty:
                        pending_result = None
                if isinstance(pending_result, tuple):
                    _close_response_handle(pending_result)
                return None

            if self._stop_event.is_set():
                if isinstance(result, tuple):
                    _close_response_handle(result)
                return None
            if isinstance(result, Exception):
                raise result
            return result

    def _interruptible_lines(self, lines: Iterable[str | bytes]) -> Iterable[str | bytes]:
        for line in lines:
            if self._stop_event.is_set():
                return
            yield line

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
        try:
            self._events.put_nowait(event)
            return True
        except queue.Full:
            # Partial deltas may be dropped under backpressure because the
            # completed event always carries the full response.
            if event["kind"] == "delta":
                return False
            if event["kind"] == "completed":
                event["deltas_dropped"] = True
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(event)
            except queue.Full:
                # Another producer may refill the queue between eviction and
                # insertion. Treat that as backpressure rather than failing.
                pass
            return False

    def _emit_available_events(self, op_output):
        # The default Holoscan output connector has capacity one. Emit one
        # event per periodic tick and leave the remainder in the bounded queue.
        try:
            event = self._events.get_nowait()
        except queue.Empty:
            return
        op_output.emit(event, "events")
