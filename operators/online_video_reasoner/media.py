# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

"""Small FFmpeg-backed encoders for RGB reasoning inputs."""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np


class MediaEncodingError(RuntimeError):
    """Raised when FFmpeg cannot encode a reasoning input."""


def validate_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Validate one HWC, 8-bit RGB frame and return it unchanged."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame shape must be HxWx3 RGB, got {frame.shape}")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame height and width must be non-zero")
    return frame


def _run_ffmpeg(command: list[str], payload: bytes, executable: str) -> bytes:
    try:
        result = subprocess.run(
            command,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise MediaEncodingError(f"cannot run FFmpeg executable {executable!r}: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = f"exit status {result.returncode}"
        raise MediaEncodingError(f"FFmpeg encoding failed: {detail}")
    return result.stdout


def encode_jpeg(frame: np.ndarray, ffmpeg: str = "ffmpeg") -> bytes:
    """Encode one RGB frame as JPEG using a single FFmpeg process."""
    validate_rgb_frame(frame)
    height, width, _ = frame.shape
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-codec:v",
        "mjpeg",
        "-q:v",
        "2",
        "-f",
        "image2pipe",
        "pipe:1",
    ]
    encoded = _run_ffmpeg(command, frame.tobytes(order="C"), ffmpeg)
    if not encoded:
        raise MediaEncodingError("FFmpeg produced an empty JPEG")
    return encoded


def encode_mp4(
    frames: Iterable[np.ndarray],
    fps: float,
    ffmpeg: str = "ffmpeg",
) -> bytes:
    """Encode ordered RGB frames as a seekable H.264 MP4."""
    try:
        frame_list = list(frames)
    except TypeError as exc:
        raise TypeError("frames must be an iterable of NumPy arrays") from exc
    if not frame_list:
        raise ValueError("frames must contain at least one RGB frame")

    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a positive finite number") from exc
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("fps must be a positive finite number")

    validate_rgb_frame(frame_list[0])
    expected_shape = frame_list[0].shape
    for index, frame in enumerate(frame_list[1:], start=1):
        validate_rgb_frame(frame)
        if frame.shape != expected_shape:
            raise ValueError(
                f"all frames must have shape {expected_shape}; frame {index} has {frame.shape}"
            )

    height, width, _ = expected_shape
    payload = b"".join(frame.tobytes(order="C") for frame in frame_list)
    with tempfile.TemporaryDirectory(prefix="online-video-reasoner-") as directory:
        output_path = Path(directory) / "clip.mp4"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps_value:.12g}",
            "-i",
            "pipe:0",
            "-frames:v",
            str(len(frame_list)),
            "-an",
            "-codec:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
        _run_ffmpeg(command, payload, ffmpeg)
        try:
            encoded = output_path.read_bytes()
        except OSError as exc:
            raise MediaEncodingError(f"cannot read encoded MP4: {exc}") from exc

    if not encoded:
        raise MediaEncodingError("FFmpeg produced an empty MP4")
    return encoded
