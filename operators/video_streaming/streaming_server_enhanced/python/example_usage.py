#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of StreamingServer Python bindings.

This example demonstrates how to use the StreamingServerUpstreamOp, 
StreamingServerDownstreamOp, and StreamingServerResource in a Python Holoscan application.
"""

import holoscan as hs
from holoscan.operators import (
    StreamingServerUpstreamOp,
    StreamingServerDownstreamOp,
    StreamingServerResource
)


class StreamingServerApp(hs.Application):
    """Example application using StreamingServer operators."""
    
    def compose(self):
        # Create shared streaming server resource
        streaming_resource = StreamingServerResource(
            self,
            width=1920,
            height=1080,
            fps=30,
            signaling_port=8554,
            streaming_port=8555,
            name="streaming_server_resource"
        )
        
        # Create upstream operator (receives from clients)
        upstream_op = StreamingServerUpstreamOp(
            self,
            width=1920,
            height=1080,
            fps=30,
            streaming_server_resource=streaming_resource,
            name="streaming_upstream"
        )
        
        # Create downstream operator (sends to clients)
        downstream_op = StreamingServerDownstreamOp(
            self,
            width=1920,
            height=1080,
            fps=30,
            enable_processing=False,
            processing_type="none",
            streaming_server_resource=streaming_resource,
            name="streaming_downstream"
        )
        
        # Add operators to workflow
        self.add_operator(upstream_op)
        self.add_operator(downstream_op)
        
        # You would typically connect these operators to your processing pipeline
        # For example:
        # upstream_op >> processing_op >> downstream_op


if __name__ == "__main__":
    app = StreamingServerApp()
    app.run()
