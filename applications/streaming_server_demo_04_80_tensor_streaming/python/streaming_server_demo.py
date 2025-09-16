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
Simple application to test the StreamingServer operators with tensor-based streaming (echo server).

.
"""

import os
import argparse
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.resources import UnboundedAllocator
from holohub.streaming_server_04_80_tensor._streaming_server_04_80_tensor import (
    StreamingServerResource,
    StreamingServerUpstreamOp, 
    StreamingServerDownstreamOp
)

class StreamingServerTestApp(Application):
    """
    Bidirectional streaming server application using Holoscan operators.
    
    This application demonstrates:
    - StreamingServerResource: Manages the underlying streaming server
    - StreamingServerUpstreamOp: Receives frames from clients
    - StreamingServerDownstreamOp: Sends frames to clients
    
    The pipeline creates an "echo server" that receives frames and sends them back.
    """

    def __init__(self, port=48010, width=854, height=480, fps=30, 
                 enable_processing=False, server_name="HoloscanStreamingServer"):
        """
        Initialize the streaming server application.
        
        Args:
            port: Port for the streaming server
            width: Frame width in pixels
            height: Frame height in pixels  
            fps: Target frames per second
            enable_processing: Whether to enable frame processing in downstream
            server_name: Name identifier for the streaming server
        """
        super().__init__()
        
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_processing = enable_processing
        self.server_name = server_name
    
    def compose(self):
        """Define the operators in the application and connect them."""
        
        # Create memory allocator for frame data
        allocator = UnboundedAllocator(self, name="allocator")
        
        # Create the shared streaming server resource
        # This manages the underlying streaming server and handles client connections
        streaming_server_resource = StreamingServerResource(
            self,
            name="streaming_server_resource",
            port=self.port,
            server_name=self.server_name,
            width=self.width,
            height=self.height, 
            fps=self.fps,
            enable_upstream=True,    # Allow receiving frames from clients
            enable_downstream=True,  # Allow sending frames to clients
            is_multi_instance=False  # Single server instance
        )
        
        # Create upstream operator - receives frames from streaming clients
        upstream_op = StreamingServerUpstreamOp(
            self,
            name="upstream_op",
            streaming_server_resource=streaming_server_resource,
            allocator=allocator
        )
        
        # Create downstream operator - sends frames to streaming clients  
        downstream_op = StreamingServerDownstreamOp(
            self,
            name="downstream_op", 
            streaming_server_resource=streaming_server_resource,
            enable_processing=self.enable_processing,  # Optional frame processing
            allocator=allocator
        )
        
        # Connect the pipeline: upstream -> downstream
        # This creates an "echo server" that receives frames and sends them back
        self.add_flow(upstream_op, downstream_op, {("output_frames", "input_frames")})
        
        print(f"🚀 Streaming Server Application Configured:")
        print(f"   📡 Server: {self.server_name}")
        print(f"   🌐 Port: {self.port}")  
        print(f"   📐 Resolution: {self.width}x{self.height}")
        print(f"   🎬 FPS: {self.fps}")
        print(f"   🔄 Mode: Echo Server (frames passed through without processing)")
        print(f"   📊 Pipeline: Upstream → Downstream (Echo Server - No Processing)")

def main():
    """Main function to parse CLI arguments and run the streaming server application."""
    parser = ArgumentParser(
        description="Holoscan Streaming Server Demo - Bidirectional Video Streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (854x480@30fps on port 48010)
  python streaming_server_demo.py
  
  # Run with custom resolution and port
  python streaming_server_demo.py --width 1920 --height 1080 --port 48020
  
  # Test with higher frame rate
  python streaming_server_demo.py --fps 60
  
  # Custom server configuration
  python streaming_server_demo.py --port 48015 --fps 60 --server-name "MyStreamingServer"

This creates a bidirectional streaming server that:
- Receives video frames from clients (upstream)
- Optionally processes the frames
- Sends frames back to clients (downstream)
        """
    )
    
    # Configuration file and data directory (for compatibility with C++ version)
    parser.add_argument("-c", "--config", type=str, default="streaming_server_demo_04_80.yaml",
                      help="Configuration file path (default: streaming_server_demo_04_80.yaml)")
    parser.add_argument("-d", "--data", type=str, default="",
                      help="Data directory (default: environment variable HOLOSCAN_INPUT_PATH or current directory)")
    
    # Server configuration
    parser.add_argument("--port", type=int, default=48010,
                      help="Port for the streaming server (default: 48010)")
    parser.add_argument("--server-name", type=str, default="HoloscanStreamingServer",
                      help="Server name identifier (default: HoloscanStreamingServer)")
    
    # Video parameters  
    parser.add_argument("--width", type=int, default=854,
                      help="Frame width in pixels (default: 854)")
    parser.add_argument("--height", type=int, default=480,
                      help="Frame height in pixels (default: 480)")
    parser.add_argument("--fps", type=int, default=30,
                      help="Target frames per second (default: 30)")
    
    # Note: --enable-processing option removed as this demo is a simple echo server without processing
    
    args = parser.parse_args()

    # Validate arguments
    if args.port < 1024 or args.port > 65535:
        print("❌ Error: Port must be between 1024 and 65535")
        return 1
        
    if args.width <= 0 or args.height <= 0:
        print("❌ Error: Width and height must be positive")
        return 1
        
    if args.fps <= 0:
        print("❌ Error: FPS must be positive") 
        return 1

    print("🐍 Holoscan Streaming Server Demo (Python)")
    print("=" * 50)
    
    try:
        # Create and run the streaming server application
        app = StreamingServerTestApp(
            port=args.port,
            width=args.width, 
            height=args.height,
            fps=args.fps,
            enable_processing=args.enable_processing,
            server_name=args.server_name
        )
        
        print("\n🎬 Starting application...")
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Application stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Application failed: {e}")
        return 1
    
    print("\n✅ Application completed successfully")
    return 0

if __name__ == "__main__":
    exit(main()) 