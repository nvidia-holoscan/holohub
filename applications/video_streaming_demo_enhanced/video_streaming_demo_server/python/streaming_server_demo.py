#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Streaming Server Demo Application (Python)

This application demonstrates how to use the StreamingServer Python bindings
to create a video streaming server that can receive and send video frames.

Features:
- Bidirectional streaming (receive from clients, send back processed frames)
- Frame processing capabilities (mirroring, filtering, etc.)
- Optional video file playback for testing
- Configurable via YAML or command-line arguments
- Multi-client support through shared resource
"""

import argparse
import os
import sys
from pathlib import Path

import holoscan as hs
from holoscan.core import Application
from holoscan.resources import UnboundedAllocator

# Import our streaming server operators
# The __init__.py automatically imports from _streaming_server_enhanced
try:
    from holohub.streaming_server_enhanced import (
        StreamingServerUpstreamOp,
        StreamingServerDownstreamOp,
        StreamingServerResource
    )
except ImportError as e:
    print(f"Error: StreamingServer operators not found: {e}")
    print("Make sure the Python bindings are built and installed.")
    print("Build with: ./holohub build streaming_server_enhanced --language cpp --configure-args='-DHOLOHUB_BUILD_PYTHON=ON'")
    sys.exit(1)


class StreamingServerApp(Application):
    """Streaming Server Demo Application using Python bindings.
    
    This matches the C++ implementation with a simple pipeline:
    StreamingServerUpstreamOp -> StreamingServerDownstreamOp
    """
    
    def __init__(self, config_file=None, port=48010, width=854, height=480, fps=30):
        super().__init__()
        self.config_file = config_file
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        
    def compose(self):
        """Compose the application pipeline.
        
        Simple bidirectional streaming:
        upstream_op (receives from clients) -> downstream_op (sends back to clients)
        """
        
        # Create allocator
        allocator = UnboundedAllocator(
            self, 
            name="allocator"
        )
        
        # Create shared streaming server resource
        streaming_resource = StreamingServerResource(
            self,
            name="streaming_server_resource",
            port=self.port,
            width=self.width,
            height=self.height,
            fps=self.fps,
            enable_upstream=True,
            enable_downstream=True
        )
        
        # Upstream operator (receives from clients)
        upstream_op = StreamingServerUpstreamOp(
            self,
            name="upstream_op",
            streaming_server_resource=streaming_resource
        )
        
        # Downstream operator (sends to clients)
        downstream_op = StreamingServerDownstreamOp(
            self,
            name="downstream_op",
            streaming_server_resource=streaming_resource
        )
        
        # Connect: upstream -> downstream
        self.add_flow(upstream_op, downstream_op, {("output_frames", "input_frames")})


def create_default_config():
    """Create a default YAML configuration file."""
    config_content = """# Streaming Server Demo Configuration
application:
  title: "Streaming Server Python Demo"
  version: "1.0"
  log_level: "INFO"

# Server configuration
streaming_server:
  port: 48010
  width: 854
  height: 480
  fps: 30
  server_ip: "127.0.0.1"
  receive_frames: true
  send_frames: true
  visualize_frames: false

# Scheduler configuration
scheduler: "greedy"
"""
    return config_content


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Streaming Server Demo (Python)")
    parser.add_argument(
        "--port",
        type=int,
        default=48010,
        help="Server port (default: 48010)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=854,
        help="Frame width (default: 854)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--create-config",
        help="Create a default configuration file at the specified path"
    )
    
    args = parser.parse_args()
    
    # Handle config file creation
    if args.create_config:
        config_path = Path(args.create_config)
        config_path.write_text(create_default_config())
        print(f"Created default configuration file: {config_path}")
        return
    
    # Create and run the application
    try:
        app = StreamingServerApp(
            config_file=args.config,
            port=args.port,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        if args.config:
            app.config(args.config)
        
        print(f"Starting Streaming Server Demo (Python)")
        print(f"Server Port: {args.port}")
        print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
        print(f"Pipeline: StreamingServerUpstreamOp -> StreamingServerDownstreamOp")
        print(f"Press Ctrl+C to stop gracefully")
        
        app.run()
        
        print("Server stopped successfully")
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
