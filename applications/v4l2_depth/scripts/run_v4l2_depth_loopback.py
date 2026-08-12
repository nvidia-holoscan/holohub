#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LOOPBACK_DEVICE = Path("/dev/video42")
LOOPBACK_WIDTH = 1280
LOOPBACK_HEIGHT = 720
LOOPBACK_FPS = 25
LOCKED_APPLICATION_OPTIONS = (
    "--data-dir",
    "--device",
    "--fps",
    "--height",
    "--validate",
    "--width",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run v4l2_depth against the managed /dev/video42 loopback stream "
            "(progressive YUYV, 1280x720 at 25 FPS, without padding)."
        )
    )
    parser.add_argument("--feeder", required=True, type=Path)
    parser.add_argument("--application", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "application_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to v4l2_depth",
    )
    return parser.parse_args()


def option_is_locked(argument: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for option in LOCKED_APPLICATION_OPTIONS
    )


def normalize_return_code(return_code: int) -> int:
    return 128 - return_code if return_code < 0 else return_code


def signal_process(process: subprocess.Popen, signal_number: int) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            pass


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    signal_process(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        signal_process(process, signal.SIGKILL)
        process.wait()


def main() -> int:
    arguments = parse_arguments()
    application_args = arguments.application_args
    if application_args[:1] == ["--"]:
        application_args = application_args[1:]

    if any(argument in ("--help", "-h") for argument in application_args):
        return subprocess.run([arguments.application, "--help"], check=False).returncode

    locked_arguments = [
        argument for argument in application_args if option_is_locked(argument)
    ]
    if locked_arguments:
        print(
            "run_v4l2_depth_loopback.py: loopback mode fixes "
            "/dev/video42 at progressive YUYV 1280x720 and 25 FPS; "
            f"remove locked option: {locked_arguments[0]}",
            file=sys.stderr,
        )
        return 2

    for path, description in (
        (arguments.feeder, "loopback feeder"),
        (arguments.application, "v4l2_depth executable"),
        (arguments.video, "loopback video"),
    ):
        if not path.is_file():
            print(
                f"run_v4l2_depth_loopback.py: {description} not found: {path}",
                file=sys.stderr,
            )
            return 1

    feeder_process = None
    application_process = None
    received_signal = None

    def forward_signal(signal_number, _frame):
        nonlocal received_signal
        received_signal = signal_number
        if application_process is not None:
            signal_process(application_process, signal_number)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)

    try:
        with tempfile.TemporaryDirectory(
            prefix="v4l2-depth-loopback-"
        ) as temporary_directory:
            ready_file = Path(temporary_directory) / "ready"
            feeder_command = [
                arguments.feeder,
                arguments.video,
                LOOPBACK_DEVICE,
                "--ready-file",
                ready_file,
            ]
            print(
                "Starting managed loopback stream: "
                f"{arguments.video} -> {LOOPBACK_DEVICE} "
                f"(progressive YUYV {LOOPBACK_WIDTH}x{LOOPBACK_HEIGHT} at {LOOPBACK_FPS} FPS)",
                flush=True,
            )
            feeder_process = subprocess.Popen(feeder_command, start_new_session=True)

            readiness_deadline = time.monotonic() + 15
            while not ready_file.is_file():
                feeder_return_code = feeder_process.poll()
                if feeder_return_code is not None:
                    print(
                        "run_v4l2_depth_loopback.py: loopback feeder exited before becoming ready",
                        file=sys.stderr,
                    )
                    return normalize_return_code(feeder_return_code) or 1
                if received_signal is not None:
                    return 128 + received_signal
                if time.monotonic() >= readiness_deadline:
                    print(
                        "run_v4l2_depth_loopback.py: timed out waiting for the loopback stream",
                        file=sys.stderr,
                    )
                    return 1
                time.sleep(0.05)

            application_command = [
                arguments.application,
                "--data-dir",
                arguments.data_dir,
                "--device",
                LOOPBACK_DEVICE,
                "--width",
                str(LOOPBACK_WIDTH),
                "--height",
                str(LOOPBACK_HEIGHT),
                "--fps",
                str(LOOPBACK_FPS),
                *application_args,
            ]
            print("Loopback stream is ready; starting v4l2_depth", flush=True)
            application_process = subprocess.Popen(
                application_command, start_new_session=True
            )

            while True:
                application_return_code = application_process.poll()
                if application_return_code is not None:
                    return normalize_return_code(application_return_code)

                feeder_return_code = feeder_process.poll()
                if feeder_return_code is not None:
                    print(
                        "run_v4l2_depth_loopback.py: loopback feeder stopped unexpectedly",
                        file=sys.stderr,
                    )
                    stop_process(application_process)
                    return normalize_return_code(feeder_return_code) or 1

                if received_signal is not None:
                    signal_process(application_process, received_signal)
                time.sleep(0.1)
    except FileNotFoundError as error:
        print(
            f"run_v4l2_depth_loopback.py: command not found: {error.filename}",
            file=sys.stderr,
        )
        return 1
    finally:
        stop_process(application_process)
        stop_process(feeder_process)


if __name__ == "__main__":
    sys.exit(main())
