# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import shutil
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

MEDIA_PATH = Path(__file__).resolve().parents[1] / "media.py"
MEDIA_SPEC = importlib.util.spec_from_file_location("online_video_reasoner_media", MEDIA_PATH)
assert MEDIA_SPEC is not None and MEDIA_SPEC.loader is not None
media = importlib.util.module_from_spec(MEDIA_SPEC)
MEDIA_SPEC.loader.exec_module(media)

MediaEncodingError = media.MediaEncodingError
encode_jpeg = media.encode_jpeg
encode_mp4 = media.encode_mp4
validate_rgb_frame = media.validate_rgb_frame


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def test_validate_rgb_frame_accepts_uint8_hwc_rgb():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    assert validate_rgb_frame(frame) is frame


@pytest.mark.parametrize(
    ("frame", "error", "message"),
    [
        ([[[0, 0, 0]]], TypeError, "NumPy array"),
        (np.zeros((4, 6, 3), dtype=np.float32), ValueError, "uint8"),
        (np.zeros((4, 6), dtype=np.uint8), ValueError, "HxWx3"),
        (np.zeros((4, 6, 4), dtype=np.uint8), ValueError, "HxWx3"),
        (np.zeros((0, 6, 3), dtype=np.uint8), ValueError, "non-zero"),
    ],
)
def test_validate_rgb_frame_rejects_invalid_input(frame, error, message):
    with pytest.raises(error, match=message):
        validate_rgb_frame(frame)


def test_encode_mp4_rejects_inconsistent_frames_before_running_ffmpeg():
    frames = [
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.zeros((5, 6, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="all frames must have shape"):
        encode_mp4(frames, 4, ffmpeg="/does/not/exist")


def test_encoder_reports_missing_ffmpeg():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    with pytest.raises(MediaEncodingError, match="cannot run FFmpeg"):
        encode_jpeg(frame, ffmpeg="/does/not/exist")


def test_encoder_reports_ffmpeg_timeout(monkeypatch):
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    process = _StalledProcess()
    moments = iter((0.0, 0.0, 61.0))

    monkeypatch.setattr(media.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(media.time, "monotonic", lambda: next(moments))

    with pytest.raises(MediaEncodingError, match="timed out after 60 seconds"):
        encode_jpeg(frame)

    assert process.communicate_calls[0][1] == 0.05
    assert process.killed
    assert process.reaped


class _StalledProcess:
    def __init__(self):
        self.returncode = None
        self.killed = False
        self.reaped = False
        self.communicate_calls = []
        self.communication_started = threading.Event()

    def communicate(self, input=None, timeout=None):
        self.communicate_calls.append((input, timeout))
        if self.killed:
            self.returncode = -9
            self.reaped = True
            return b"", b""
        self.communication_started.set()
        raise subprocess.TimeoutExpired("ffmpeg", timeout)

    def kill(self):
        self.killed = True


def test_encoder_cancellation_kills_and_reaps_ffmpeg(monkeypatch):
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    cancel_event = threading.Event()
    process = _StalledProcess()
    errors = []

    monkeypatch.setattr(media.subprocess, "Popen", lambda *args, **kwargs: process)

    def encode():
        try:
            encode_jpeg(frame, cancel_event=cancel_event)
        except MediaEncodingError as error:
            errors.append(error)

    worker = threading.Thread(target=encode)
    worker.start()
    assert process.communication_started.wait(timeout=1)
    cancel_event.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], MediaEncodingError)
    assert str(errors[0]) == "FFmpeg encoding cancelled"
    assert process.killed
    assert process.reaped


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
def test_encode_jpeg_has_jpeg_signature():
    frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    encoded = encode_jpeg(frame, ffmpeg=FFMPEG)
    assert encoded.startswith(b"\xff\xd8")
    assert encoded.endswith(b"\xff\xd9")


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required",
)
def test_encode_mp4_preserves_timing_and_frame_order(tmp_path):
    fps = 5
    colours = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0))
    frames = [np.full((12, 16, 3), colour, dtype=np.uint8) for colour in colours]
    encoded = encode_mp4(frames, fps=fps, ffmpeg=FFMPEG)
    assert encoded[4:8] == b"ftyp"

    output_path = tmp_path / "clip.mp4"
    output_path.write_bytes(encoded)
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=avg_frame_rate,nb_read_frames,duration:format=format_name",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    assert int(stream["nb_read_frames"]) == len(frames)
    assert Fraction(stream["avg_frame_rate"]) == Fraction(fps, 1)
    assert float(stream["duration"]) == pytest.approx(len(frames) / fps, abs=0.02)
    assert "mp4" in metadata["format"]["format_name"].split(",")

    decoded = subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(output_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    decoded_frames = np.frombuffer(decoded, dtype=np.uint8).reshape((-1, 12, 16, 3))
    dominant_channels = [int(frame.mean(axis=(0, 1)).argmax()) for frame in decoded_frames]
    assert dominant_channels == [0, 1, 2, 0]
