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
Simple application to test the StreamingClientOp operator.
"""

import os
import argparse
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import StreamingClientOp
from holoscan.resources import UnboundedAllocator

class StreamingClientTestApp(Application):
    """Example application to test the StreamingClientOp operator."""

    def __init__(self, server_ip, signaling_port=48010, width=1920, height=1080, fps=30):
        """Initialize the application."""
        super().__init__()
        
        # Get the directory where this file is located
        self.server_ip = server_ip
        self.signaling_port = signaling_port
        self.width = width
        self.height = height
        self.fps = fps
    
    def compose(self):
        """Define the operators in the application and connect them."""
        # Define resources and operators
        unbounded_allocator = UnboundedAllocator(self, name="allocator")
        
        # Create the streaming client operator
        streaming_client = StreamingClientOp(
            self,
            name="streaming_client",
            allocator=unbounded_allocator,
            width=self.width,
            height=self.height,
            fps=self.fps,
            server_ip=self.server_ip,
            signaling_port=self.signaling_port,
            receive_frames=True,
            send_frames=False,
        )
        
        # Add the operators to the application
        self.add_operator(streaming_client)

def main():
    """Main function to parse CLI arguments and run the application."""
    parser = ArgumentParser(description="Streaming Client Test Application")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1",
                      help="IP address of the streaming server")
    parser.add_argument("--signaling_port", type=int, default=48010,
                      help="Port for signaling")
    parser.add_argument("--width", type=int, default=1920,
                      help="Frame width")
    parser.add_argument("--height", type=int, default=1080,
                      help="Frame height")
    parser.add_argument("--fps", type=int, default=30,
                      help="Frames per second")
    args = parser.parse_args()

    # Create and run the application
    app = StreamingClientTestApp(
        server_ip=args.server_ip,
        signaling_port=args.signaling_port,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    app.run()

if __name__ == "__main__":
    main() 