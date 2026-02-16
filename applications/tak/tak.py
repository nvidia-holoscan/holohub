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

    def __init__(self, video_dir, data, source="replayer"):
        super().__init__()
        self.name = "TAK Detection App"
        self.source = source

        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOHUB_DATA_PATH", "../data"), "tak"
            )
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
        bytetrack_config = detector_kwargs.get(
            "bytetrack_config", "bytetrack.yaml")
        bytetrack_path = str(Path(__file__).with_name(bytetrack_config))
        data_model = os.path.join(self.data, model_path)
        if os.path.exists(data_model):
            detector_kwargs["model_path"] = data_model

        detector = DetectorOp(
            self,
            name="detector",
            bytetrack_path=bytetrack_path,
            **detector_kwargs,
        )

        tak_cot = TakCotOp(
            self,
            name="tak_cot",
            marker_type="a-h-G",
            detector_op=detector,
            **self.kwargs("tak_cot"),
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
    parser = ArgumentParser(
        description="TAK: YOLOv8 Detection with TAK Integration.")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "tak.yaml"),
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-s", "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help="Input source: 'v4l2' for V4L2 device or 'replayer' for video stream replayer.",
    )
    parser.add_argument(
        "-d", "--data",
        default="none",
        help="Path to the data directory (model + video).",
    )
    parser.add_argument(
        "-v", "--video_dir",
        default="none",
        help="Path to the video directory (for replayer mode).",
    )

    args = parser.parse_args()

    # Logging setup
    log_level_str = os.getenv("HOLOSCAN_LOG_LEVEL", "INFO").upper()
    python_log_map = {
        "TRACE": logging.DEBUG, "DEBUG": logging.DEBUG,
        "INFO": logging.INFO, "WARN": logging.WARNING,
        "WARNING": logging.WARNING, "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig(
        level=python_log_map.get(log_level_str, logging.INFO),
        format="[%(levelname)s] [%(name)s] %(message)s",
    )
    holoscan_log_map = {
        "TRACE": LogLevel.TRACE, "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO, "WARN": LogLevel.WARN,
        "WARNING": LogLevel.WARN, "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
    }
    set_log_level(holoscan_log_map.get(log_level_str, LogLevel.INFO))
    set_log_pattern("[%^%l%$] [%n] %v")

    # TAK server config from environment
    tak_host_raw = os.getenv("TAK_HOST", "")
    if tak_host_raw.startswith("http://") or tak_host_raw.startswith("https://"):
        from urllib.parse import urlparse
        tak_host = urlparse(tak_host_raw).hostname
    else:
        tak_host = tak_host_raw

    # Start OpenTAKServer services if the startup script exists
    ots_script = "/opt/ots/start_ots.sh"
    if os.path.isfile(ots_script):
        import subprocess
        logging.getLogger("tak").info("Starting OpenTAKServer services...")
        subprocess.Popen(
            ["bash", ots_script],
            stdout=open("/tmp/ots_start.log", "w"),
            stderr=subprocess.STDOUT,
        )

    app = TAKApp(
        video_dir=args.video_dir,
        data=args.data,
        source=args.source,
    )
    app.config(args.config)
    app.run()


if __name__ == "__main__":
    main()
