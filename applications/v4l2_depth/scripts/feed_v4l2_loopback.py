#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

LOOPBACK_WIDTH = 1280
LOOPBACK_HEIGHT = 720
LOOPBACK_FPS = 25


def parse_arguments() -> argparse.Namespace:
    repository_directory = Path(__file__).resolve().parents[3]
    default_video = (
        repository_directory / "data" / "v4l2_depth" / "5823544-hd_1920_1080_25fps.mp4"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Loop an H.264 MP4 into a V4L2 output device as "
            f"{LOOPBACK_WIDTH}x{LOOPBACK_HEIGHT} progressive YUYV at "
            f"{LOOPBACK_FPS} FPS without padding."
        )
    )
    parser.add_argument("video_file", nargs="?", type=Path, default=default_video)
    parser.add_argument(
        "video_device", nargs="?", type=Path, default=Path("/dev/video42")
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Touch this file after the GStreamer pipeline reaches PLAYING",
    )
    return parser.parse_args()


def quoted_gstreamer_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> int:
    arguments = parse_arguments()
    video_file = arguments.video_file.resolve()
    video_device = arguments.video_device.resolve()

    if not video_file.is_file():
        print(
            f"feed_v4l2_loopback.py: video file not found: {video_file}",
            file=sys.stderr,
        )
        return 1
    try:
        device_mode = video_device.stat().st_mode
    except FileNotFoundError:
        print(
            f"feed_v4l2_loopback.py: V4L2 device not found: {video_device}",
            file=sys.stderr,
        )
        return 1
    if not stat.S_ISCHR(device_mode):
        print(
            f"feed_v4l2_loopback.py: V4L2 device is not a character device: {video_device}",
            file=sys.stderr,
        )
        return 1
    if not os.access(video_device, os.R_OK | os.W_OK):
        print(
            f"feed_v4l2_loopback.py: read/write permission required: {video_device}",
            file=sys.stderr,
        )
        return 1

    try:
        subprocess.run(
            [
                "v4l2-ctl",
                f"--device={video_device}",
                (
                    f"--set-fmt-video-out=width={LOOPBACK_WIDTH},"
                    f"height={LOOPBACK_HEIGHT},pixelformat=YUYV,field=none"
                ),
            ],
            check=True,
        )
        subprocess.run(
            [
                "v4l2-ctl",
                f"--device={video_device}",
                f"--set-parm={LOOPBACK_FPS}",
            ],
            check=True,
        )
    except FileNotFoundError:
        print(
            "feed_v4l2_loopback.py: required command not found: v4l2-ctl",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as error:
        print(f"feed_v4l2_loopback.py: v4l2-ctl failed: {error}", file=sys.stderr)
        return 1

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError) as error:
        print(
            f"feed_v4l2_loopback.py: GStreamer Python bindings are unavailable: {error}",
            file=sys.stderr,
        )
        return 1

    Gst.init(None)
    required_elements = (
        "avdec_h264",
        "filesrc",
        "qtdemux",
        "videoconvert",
        "videoscale",
        "v4l2sink",
    )
    for element in required_elements:
        if Gst.ElementFactory.find(element) is None:
            print(
                f"feed_v4l2_loopback.py: required GStreamer element not found: {element}",
                file=sys.stderr,
            )
            return 1

    description = " ! ".join(
        (
            f"filesrc location={quoted_gstreamer_value(str(video_file))}",
            "qtdemux",
            "avdec_h264",
            "videoconvert",
            "videoscale add-borders=false",
            (
                f"video/x-raw,format=YUY2,width={LOOPBACK_WIDTH},"
                f"height={LOOPBACK_HEIGHT},framerate={LOOPBACK_FPS}/1,"
                "pixel-aspect-ratio=1/1"
            ),
            (
                f"v4l2sink device={quoted_gstreamer_value(str(video_device))} "
                "io-mode=mmap sync=true"
            ),
        )
    )
    try:
        pipeline = Gst.parse_launch(description)
    except (
        Exception
    ) as error:  # Gst raises a dynamically generated GLib.Error subclass.
        print(
            f"feed_v4l2_loopback.py: creating GStreamer pipeline failed: {error}",
            file=sys.stderr,
        )
        return 1

    stop_requested = False

    def request_stop(_signal_number, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        f"Streaming {video_file} to {video_device} as "
        f"{LOOPBACK_WIDTH}x{LOOPBACK_HEIGHT} progressive YUYV at "
        f"{LOOPBACK_FPS} FPS without padding",
        flush=True,
    )
    print(
        "Playback seeks to the first frame at end-of-file; press Ctrl-C to stop.",
        flush=True,
    )

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print(
            "feed_v4l2_loopback.py: starting GStreamer pipeline failed", file=sys.stderr
        )
        pipeline.set_state(Gst.State.NULL)
        return 1

    state_change, current_state, _pending_state = pipeline.get_state(10 * Gst.SECOND)
    if (
        state_change == Gst.StateChangeReturn.FAILURE
        or current_state != Gst.State.PLAYING
    ):
        print(
            "feed_v4l2_loopback.py: GStreamer pipeline did not reach PLAYING",
            file=sys.stderr,
        )
        pipeline.set_state(Gst.State.NULL)
        return 1
    if arguments.ready_file is not None:
        arguments.ready_file.parent.mkdir(parents=True, exist_ok=True)
        arguments.ready_file.touch()
    print("Loopback stream ready", flush=True)

    result = 0
    bus = pipeline.get_bus()
    message_types = Gst.MessageType.ERROR | Gst.MessageType.EOS
    try:
        while not stop_requested:
            message = bus.timed_pop_filtered(250 * Gst.MSECOND, message_types)
            if message is None:
                continue
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                print(
                    f"feed_v4l2_loopback.py: GStreamer error: {error}", file=sys.stderr
                )
                if debug:
                    print(debug, file=sys.stderr)
                result = 1
                break
            if message.type == Gst.MessageType.EOS:
                seek_flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT
                if not pipeline.seek_simple(Gst.Format.TIME, seek_flags, 0):
                    print(
                        "feed_v4l2_loopback.py: seeking to the first frame failed",
                        file=sys.stderr,
                    )
                    result = 1
                    break
                print("Restarted video at end-of-file", flush=True)
    finally:
        pipeline.set_state(Gst.State.NULL)

    return result


if __name__ == "__main__":
    sys.exit(main())
