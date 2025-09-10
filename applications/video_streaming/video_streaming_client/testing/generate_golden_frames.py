#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate golden reference frames for streaming_client_demo testing.
This runs the streaming client demo with recording enabled to capture reference frames.
"""

import os
import sys
from argparse import ArgumentParser

# Add the parent directory to sys.path to import the streaming_client_demo module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from holoscan.core import Application
from holoscan.operators import FormatConverterOp, VideoStreamRecorderOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool


class StreamingClientTestAppWithRecording(Application):
    """Streaming client demo with video recording for generating golden frames."""

    def __init__(
        self,
        width=854,
        height=480,
        fps=30,
        data_path=None,
        recording_dir="./recording_output",
        recording_basename="streaming_client_demo_output",
        frame_count=10,
    ):
        """Initialize the application with recording capability."""
        super().__init__()

        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.recording_dir = recording_dir
        self.recording_basename = recording_basename

        # Search for video data in common locations
        if data_path is None:
            search_paths = [
                os.environ.get("HOLOSCAN_DATA_PATH", ""),
                "/workspace/holohub/data/endoscopy",
                "./data/endoscopy",
                "../data/endoscopy",
                "../../data/endoscopy",
            ]

            for path in search_paths:
                if path and os.path.exists(os.path.join(path, "surgical_video.gxf_index")):
                    data_path = path
                    print(f"Found video data at: {data_path}")
                    break

            if data_path is None:
                print("Warning: surgical_video.gxf_index not found in standard locations.")
                print("Using default path - video may not load correctly.")
                data_path = "./data/endoscopy"

        self.data_path = data_path

    def compose(self):
        # Create resources
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Source: Video replayer
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.data_path,
            basename="surgical_video",
            frame_rate=self.fps,
            repeat=False,  # Don't repeat for testing
            realtime=False,  # Run as fast as possible for testing
            count=self.frame_count,  # Limit to specific number of frames
        )

        # Format converter
        source_block_size = self.width * self.height * 3
        source_num_blocks = 4
        pool = BlockMemoryPool(
            self,
            name="pool",
            storage_type=1,
            block_size=source_block_size,
            num_blocks=source_num_blocks,
        )

        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=pool,
            out_tensor_name="tensor",
            out_dtype="uint8",
            cuda_stream_pool=cuda_stream_pool,
        )

        # Video recorder to capture golden frames
        recorder = VideoStreamRecorderOp(
            self,
            name="recorder",
            directory=self.recording_dir,
            basename=self.recording_basename,
            record_type="visualizer",  # Record visualizer output
        )

        # Connect the pipeline: source -> format_converter -> recorder
        self.add_flow(source, format_converter)
        self.add_flow(format_converter, recorder, {("tensor", "input")})

        print(f"Recording {self.frame_count} frames to {self.recording_dir}")


def main():
    parser = ArgumentParser(
        description="Generate golden reference frames for streaming_client_demo testing"
    )
    parser.add_argument("--data", default=None, help="Path to video data directory")
    parser.add_argument(
        "--output-dir", default="./recording_output", help="Output directory for recorded frames"
    )
    parser.add_argument(
        "--basename", default="streaming_client_demo_output", help="Basename for recorded files"
    )
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to record")
    parser.add_argument("--width", type=int, default=854, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create and run the application
    app = StreamingClientTestAppWithRecording(
        width=args.width,
        height=args.height,
        fps=args.fps,
        data_path=args.data,
        recording_dir=args.output_dir,
        recording_basename=args.basename,
        frame_count=args.frames,
    )

    app.run()
    print(f"\nRecording complete! Check {args.output_dir} for output files.")
    print("These files can be used as golden reference frames for testing.")


if __name__ == "__main__":
    main()
