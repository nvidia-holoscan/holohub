# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TAK: Real-time YOLOv8 object detection with TAK server integration.

Runs YOLOv8 detection with ByteTrack tracking on video from either a
V4L2 camera or a pre-recorded replayer source, visualizes results via
Holoviz, and uploads Cursor-on-Target markers to a TAK server.
"""

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from holoscan.core import Application
from holoscan.logger import LogLevel, set_log_level, set_log_pattern
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator
from operators.detector_op import DetectorOp
from operators.tak_cot_op import TakCotOp


class TAKApp(Application):
    """YOLOv8 detection with TAK CoT marker upload."""

    def __init__(self, video_dir, data, source="replayer", tak_host_override=None):
        super().__init__()
        self.name = "TAK Detection App"
        self.source = source
        self.tak_host_override = tak_host_override

        if data == "none":
            default_data = os.environ.get(
                "HOLOHUB_DATA_PATH",
                str(Path(__file__).resolve().parent.parent.parent / "data"),
            )
            data = os.path.join(default_data, "tak")
        self.data = data

        if video_dir == "none":
            video_dir = data
        self.video_dir = video_dir

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        # Input source (replayer or V4L2 camera)
        if self.source == "v4l2":
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **self.kwargs("v4l2_source"),
            )
            source_output = "signal"
            in_dtype = "rgba8888"
        else:
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=self.video_dir,
                **self.kwargs("replayer"),
            )
            source_output = "output"
            in_dtype = "rgb888"

        preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("detection_preprocessor"),
        )

        # Resolve model path from config against the data directory
        detector_kwargs = self.kwargs("detector")
        model_path = detector_kwargs.get("model_path", "yolov8s.pt")
        bytetrack_config = detector_kwargs.pop("bytetrack_path", "bytetrack.yaml")
        bt_cfg_path = Path(bytetrack_config)
        if bt_cfg_path.is_absolute():
            bytetrack_path = str(bt_cfg_path)
        else:
            bytetrack_path = str(Path(__file__).resolve().parent / bytetrack_config)
        if not os.path.isfile(bytetrack_path):
            raise FileNotFoundError(f"ByteTrack config not found at {bytetrack_path}")
        data_model = os.path.join(self.data, model_path)
        if os.path.exists(data_model):
            detector_kwargs["model_path"] = data_model

        detector = DetectorOp(
            self,
            name="detector",
            bytetrack_path=bytetrack_path,
            **detector_kwargs,
        )

        tak_kwargs = dict(self.kwargs("tak_cot"))
        if self.tak_host_override is not None:
            tak_kwargs["tak_host"] = self.tak_host_override

        tak_cot = TakCotOp(
            self,
            name="tak_cot",
            marker_type="a-h-G",
            marker_type_map={
                "person": "a-h-G-U-C",
                "car": "a-h-G-E-V-C",
                "truck": "a-h-G-E-V-T",
                "bus": "a-h-G-E-V-U",
                "motorcycle": "a-h-G-E-V-M",
                "bicycle": "a-n-G",
            },
            detector_op=detector,
            **tak_kwargs,
        )

        holoviz = HolovizOp(
            self,
            name="holoviz",
            tensors=[
                dict(name="", type="color"),
            ],
            window_close_callback=self.on_window_closed,
            **self.kwargs("holoviz"),
        )

        # Data flow
        self.add_flow(source, holoviz, {(source_output, "receivers")})
        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, detector, {("", "in")})
        self.add_flow(detector, holoviz, {("outputs", "receivers")})
        self.add_flow(detector, holoviz, {("output_specs", "input_specs")})
        self.add_flow(detector, tak_cot, {("tak_out", "in")})

    def on_window_closed(self):
        self.stop_execution()


def main():
    parser = ArgumentParser(description="TAK: YOLOv8 Detection with TAK Integration.")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "tak.yaml"),
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help="Input source: 'v4l2' for V4L2 device or 'replayer' for video stream replayer.",
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help="Path to the data directory (model + video).",
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        default="none",
        help="Path to the video directory (for replayer mode).",
    )

    args = parser.parse_args()

    # Logging setup
    log_level_str = os.getenv("HOLOSCAN_LOG_LEVEL", "INFO").upper()
    python_log_map = {
        "TRACE": logging.DEBUG,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig(
        level=python_log_map.get(log_level_str, logging.INFO),
        format="[%(levelname)s] [%(name)s] %(message)s",
    )
    holoscan_log_map = {
        "TRACE": LogLevel.TRACE,
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARN": LogLevel.WARN,
        "WARNING": LogLevel.WARN,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
    }
    set_log_level(holoscan_log_map.get(log_level_str, LogLevel.INFO))
    set_log_pattern("[%^%l%$] [%n] %v")

    # TAK server config: merge environment override with tak.yaml values
    tak_host_override = None
    tak_host_raw = os.getenv("TAK_HOST")
    if tak_host_raw is not None:
        if tak_host_raw.startswith("http://") or tak_host_raw.startswith("https://"):
            from urllib.parse import urlparse

            tak_host_override = urlparse(tak_host_raw).hostname or ""
        else:
            tak_host_override = tak_host_raw

    # Determine the effective TAK host and port from env override or tak.yaml
    effective_host = tak_host_override
    effective_port = None
    try:
        import yaml

        with open(args.config, "r") as _cf:
            _cfg = yaml.safe_load(_cf) or {}
        tak_cot_cfg = _cfg.get("tak_cot") or {}
        if effective_host is None:
            effective_host = tak_cot_cfg.get("tak_host", "localhost")
        effective_port = tak_cot_cfg.get("tak_port")
    except Exception:
        if effective_host is None:
            effective_host = "localhost"

    # Start OpenTAKServer services only when the target is local and not empty
    # (empty host means TAK is disabled — TakCotOp skips connection)
    _local_hosts = {"localhost", "127.0.0.1", "::1"}
    ots_script = "/opt/ots/start_ots.sh"
    if os.path.isfile(ots_script) and effective_host in _local_hosts:
        import socket
        import subprocess
        import time

        tak_logger = logging.getLogger("tak")
        first_run = not os.path.isfile("/opt/ots/.setup_complete")
        if first_run:
            tak_logger.info(
                "First run: OpenTAKServer will be downloaded and installed. "
                "This may take 1-2 minutes. Subsequent launches will be faster."
            )
        # Export configured port so start_ots.sh and nginx use it
        if effective_port is not None:
            os.environ.setdefault("OTS_COT_PORT", str(effective_port))

        tak_logger.info("Starting OpenTAKServer services...")
        ots_log = open("/tmp/ots_start.log", "w")
        ots_proc = subprocess.Popen(
            ["bash", "-u", ots_script],
            stdout=ots_log,
            stderr=subprocess.STDOUT,
        )
        ots_log.close()
        # Tail the log in a background thread so OTS progress is visible
        import threading

        stop_tail = threading.Event()

        def _tail_ots_log():
            with open("/tmp/ots_start.log", "r") as f:
                while not stop_tail.is_set():
                    line = f.readline()
                    if line:
                        tak_logger.info(line.rstrip())
                    else:
                        stop_tail.wait(0.2)

        log_thread = threading.Thread(target=_tail_ots_log, daemon=True)
        log_thread.start()

        # Wait for the TCP CoT port to be accepting connections
        cot_port = int(os.getenv("OTS_COT_PORT", "18088"))
        tak_logger.info("Waiting for OTS to be ready (TCP port %d)...", cot_port)
        start_wait = time.time()
        for attempt in range(60):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect(("localhost", cot_port))
                elapsed = time.time() - start_wait
                tak_logger.info("OTS is ready (took %.0fs)", elapsed)
                break
            except OSError:
                if ots_proc.poll() is not None and ots_proc.returncode != 0:
                    tak_logger.error(
                        "start_ots.sh exited with code %d before OTS became ready. "
                        "Check /tmp/ots_start.log for details.",
                        ots_proc.returncode,
                    )
                    break
                elapsed = time.time() - start_wait
                if attempt > 0 and attempt % 5 == 0:
                    tak_logger.info("Still waiting for OTS... (%.0fs elapsed)", elapsed)
                time.sleep(2)
        else:
            tak_logger.warning(
                "OTS did not become ready within 120s; "
                "app will start anyway (TAK integration may not work). "
                "Check /tmp/ots_start.log for details."
            )

        stop_tail.set()
        log_thread.join(timeout=1.0)

    app = TAKApp(
        video_dir=args.video_dir,
        data=args.data,
        source=args.source,
        tak_host_override=tak_host_override,
    )
    app.config(args.config)
    app.run()


if __name__ == "__main__":
    main()
