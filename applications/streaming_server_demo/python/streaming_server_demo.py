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
Simple application to test the StreamingServerOp operator.
"""

from argparse import ArgumentParser

from holoscan.core import Application

from holohub.streaming_server import StreamingServerOp


class StreamingServerTestApp(Application):
    """Example application to test the StreamingServerOp operator."""

    def __init__(self, port=8080, width=1920, height=1080, fps=30):
        """Initialize the application."""
        super().__init__()

        # Server configuration
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps

    def compose(self):
        """Define the operators in the application and connect them."""
        # Create the streaming server operator
        streaming_server = StreamingServerOp(
            self,
            name="streaming_server",
        )

        # Add the operators to the application
        self.add_operator(streaming_server)


def main():
    """Main function to parse CLI arguments and run the application."""
    parser = ArgumentParser(description="Streaming Server Demo Application")
    parser.add_argument("--port", type=int, default=8080, help="Port for streaming server")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    # Create and run the application
    app = StreamingServerTestApp(
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    app.run()


if __name__ == "__main__":
    main()
