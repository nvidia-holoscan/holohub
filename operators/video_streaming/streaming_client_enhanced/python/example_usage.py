#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of StreamingClientOp Python bindings.

This example demonstrates how to use the StreamingClientOp in a Python Holoscan application.
"""

import holoscan as hs
from holoscan.operators import StreamingClientOp


class StreamingClientApp(hs.Application):
    """Example application using StreamingClientOp."""
    
    def compose(self):
        # Create streaming client operator
        streaming_client = StreamingClientOp(
            self,
            width=1920,
            height=1080,
            fps=30,
            server_ip="127.0.0.1",
            signaling_port=8554,
            send_frames=True,
            receive_frames=False,
            min_non_zero_bytes=100,
            name="streaming_client"
        )
        
        # Add to workflow (you would typically connect this to other operators)
        self.add_operator(streaming_client)


if __name__ == "__main__":
    app = StreamingClientApp()
    app.run()
