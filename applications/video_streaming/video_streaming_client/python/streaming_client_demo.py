#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Streaming Client Demo Application (Python)

This application demonstrates how to use the StreamingClientOp Python bindings
to create a video streaming client that can send and receive video frames.

Features:
- Video file playback using VideoStreamReplayer
- V4L2 camera capture support
- Format conversion (RGB to BGR)
- Real-time video streaming to server
- Optional visualization with Holoviz
- Configurable via YAML or command-line arguments
"""

import argparse
import os
import sys
from pathlib import Path

from holoscan.core import Application
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import CudaStreamPool, UnboundedAllocator

# Import our streaming client operator
# The __init__.py automatically imports from _streaming_client_enhanced
try:
    from holohub.streaming_client_enhanced import StreamingClientOp
except ImportError as e:
    print(f"Error: StreamingClientOp not found: {e}")
    print("Make sure the Python bindings are built and installed.")
    print(
        "Build with: ./holohub build streaming_client_enhanced --language cpp --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'"
    )
    sys.exit(1)


class StreamingClientApp(Application):
    """Streaming Client Demo Application using Python bindings."""

    def __init__(
        self,
        source="replayer",
        server_ip="127.0.0.1",
        port=48010,
        width=854,
        height=480,
        fps=30,
        visualize=True,
    ):
        super().__init__()
        self.source = source
        self.server_ip = server_ip
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.visualize = visualize

    def compose(self):
        """Compose the application pipeline."""

        # Create allocator and CUDA stream pool
        allocator = UnboundedAllocator(self, name="allocator")

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Create source operator based on configuration
        if self.source == "replayer":
            source_op = self._create_replayer_source(allocator)
            # VideoStreamReplayer outputs RGB (3 channels)
            in_dtype = "rgb888"
        elif self.source == "v4l2":
            source_op = self._create_v4l2_source(allocator)
            # V4L2 outputs RGBA (4 channels) after internal YUYV->RGBA conversion
            in_dtype = "rgba8888"
        else:
            raise ValueError(f"Unknown source type: {self.source}")

        # Create format converter with correct input dtype based on source
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            in_dtype=in_dtype,  # Conditional: rgba8888 for V4L2, rgb888 for replayer
            out_dtype="rgb888",  # Always output RGB (3 channels), dropping alpha if present
            out_tensor_name="tensor",
            scale_min=0.0,
            scale_max=255.0,
            out_channel_order=[2, 1, 0],  # Convert RGB to BGR
            pool=allocator,
            cuda_stream_pool=cuda_stream_pool,
        )

        # Create streaming client (allocator added via positional args if supported)
        streaming_client = StreamingClientOp(
            self,
            allocator,  # Add allocator for output buffer
            name="streaming_client",
            width=self.width,
            height=self.height,
            fps=self.fps,
            server_ip=self.server_ip,
            signaling_port=self.port,
            send_frames=True,
            receive_frames=True,
            min_non_zero_bytes=10,
        )

        # Connect the pipeline: source -> format_converter -> streaming_client -> holoviz
        # Match C++ implementation
        if self.source == "v4l2":
            # V4L2 outputs on "signal" port
            self.add_flow(source_op, format_converter, {("signal", "source_video")})
        else:
            # VideoStreamReplayer outputs on "output" port
            self.add_flow(source_op, format_converter, {("output", "source_video")})

        # Format converter output -> streaming client input
        self.add_flow(format_converter, streaming_client)

        # Optional visualization
        if self.visualize:
            holoviz = HolovizOp(
                self,
                name="holoviz",
                width=self.width,
                height=self.height,
                allocator=allocator,
                cuda_stream_pool=cuda_stream_pool,
                tensors=[
                    {
                        "name": "bgra_tensor",  # Tensor name to match streaming client output
                        "type": "color",
                        "image_format": "b8g8r8a8_unorm",  # BGRA format (4 channels)
                        "opacity": 1.0,
                        "priority": 0,
                    }
                ],
            )

            # Connect streaming client output -> holoviz (receives frames from server)
            # Explicitly map output_frames port to receivers port (matching C++ implementation)
            self.add_flow(streaming_client, holoviz, {("output_frames", "receivers")})

    def _create_replayer_source(self, allocator):
        """Create video stream replayer source."""
        # Default to endoscopy data if available
        data_path = os.environ.get("HOLOHUB_DATA_PATH", "/workspace/holohub/data")
        video_dir = os.path.join(data_path, "endoscopy")

        if not os.path.exists(video_dir):
            print(f"Warning: Video directory not found at {video_dir}")
            print("Using current directory for video files")
            video_dir = "."

        return VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            basename="surgical_video",
            frame_rate=self.fps,
            repeat=True,
            realtime=True,
            count=0,
            allocator=allocator,
        )

    def _create_v4l2_source(self, allocator):
        """Create V4L2 camera capture source."""
        return V4L2VideoCaptureOp(
            self,
            name="v4l2_source",
            device="/dev/video0",
            width=self.width,
            height=self.height,
            frame_rate=self.fps,
            pixel_format="YUYV",
            allocator=allocator,
        )


def create_default_config():
    """Create a default YAML configuration file."""
    config_content = """# Streaming Client Demo Configuration
application:
  title: "Streaming Client Python Demo"
  version: "1.0"
  log_level: "INFO"

# Source configuration
source: "replayer"  # Options: "replayer", "v4l2"

# Video replayer settings
replayer:
  directory: "/workspace/holohub/data/endoscopy"
  basename: "surgical_video"
  frame_rate: 30
  repeat: true
  realtime: true

# V4L2 camera settings
v4l2:
  device: "/dev/video0"
  width: 640
  height: 480
  frame_rate: 30
  pixel_format: "YUYV"

# Streaming client settings
streaming_client:
  width: 854
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  signaling_port: 48010
  send_frames: true
  receive_frames: true
  min_non_zero_bytes: 10

# Visualization settings
visualization:
  enabled: true
  width: 854
  height: 480
"""
    return config_content


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Streaming Client Demo (Python)")
    parser.add_argument(
        "--source",
        choices=["replayer", "v4l2"],
        default="replayer",
        help="Video source type (default: replayer)",
    )
    parser.add_argument("--config", "-c", help="Path to YAML configuration file")
    parser.add_argument(
        "--server-ip", default="127.0.0.1", help="Server IP address (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=48010, help="Server port (default: 48010)")
    parser.add_argument("--width", type=int, default=854, help="Frame width (default: 854)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument(
        "--create-config", help="Create a default configuration file at the specified path"
    )

    args = parser.parse_args()

    # Handle config file creation
    if args.create_config:
        config_path = Path(args.create_config)
        config_path.write_text(create_default_config())
        print(f"Created default configuration file: {config_path}")
        return

    # Auto-select config file based on source if not specified
    config_file = args.config
    if not config_file:
        if args.source == "replayer":
            config_file = "streaming_client_demo_replayer.yaml"
        else:  # v4l2
            config_file = "streaming_client_demo.yaml"
        print(f"Auto-selected config file for {args.source}: {config_file}")

    # Create and run the application
    try:
        # Start with default parameters
        app = StreamingClientApp(
            source="replayer",
            server_ip="127.0.0.1",
            port=48010,
            width=854,
            height=480,
            fps=30,
            visualize=True,
        )

        # Load config file if provided (overrides defaults)
        if config_file:
            app.config(config_file)

        # Apply command-line arguments (overrides config file)
        # Only override if explicitly provided
        if args.source != "replayer":
            app.source = args.source
        if args.server_ip != "127.0.0.1":
            app.server_ip = args.server_ip
        if args.port != 48010:
            app.port = args.port
        if args.width != 854:
            app.width = args.width
        if args.height != 480:
            app.height = args.height
        if args.fps != 30:
            app.fps = args.fps
        if args.no_viz:
            app.visualize = False

        print("Starting Streaming Client Demo (Python)")
        print(f"Source: {app.source}")
        print(f"Server: {app.server_ip}:{app.port}")
        print(f"Resolution: {app.width}x{app.height} @ {app.fps}fps")
        print(f"Visualization: {'enabled' if app.visualize else 'disabled'}")

        # Show pipeline
        viz_part = " -> HolovizOp" if app.visualize else ""
        print(
            f"Pipeline: {app.source.capitalize()}Op -> FormatConverterOp -> StreamingClientOp{viz_part}"
        )
        print("Press Ctrl+C to stop gracefully")

        app.run()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
