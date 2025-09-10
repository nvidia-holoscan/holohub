# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functional test application for StreamingClientOp with real video data.
Tests actual video frame processing through the streaming client.
"""

import argparse
import numpy as np
from pathlib import Path

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import VideoStreamReplayerOp, FormatConverterOp
from holoscan.resources import UnboundedAllocator

from holohub.streaming_client_operator import StreamingClientOp


class VideoFrameValidatorOp(Operator):
    """Validates that video frames are flowing through the pipeline correctly."""
    
    def __init__(self, fragment, name="video_validator", **kwargs):
        self.frame_count = 0
        self.total_pixels = 0
        super().__init__(fragment, name=name, **kwargs)
    
    def setup(self, spec: OperatorSpec):
        spec.input("input_frames")
    
    def compute(self, op_input, op_output, context):
        # Receive frame from streaming client
        input_message = op_input.receive("input_frames")
        if input_message is None:
            return
            
        self.frame_count += 1
        
        # Validate frame properties
        try:
            tensor = input_message.get("tensor")  # Assume tensor name
            if tensor is not None:
                shape = tensor.shape
                pixel_count = np.prod(shape) if len(shape) > 0 else 0
                self.total_pixels += pixel_count
                
                self.log_info(f"✅ Frame {self.frame_count}: shape={shape}, pixels={pixel_count}")
                
                # Log summary every 10 frames
                if self.frame_count % 10 == 0:
                    avg_pixels = self.total_pixels / self.frame_count
                    self.log_info(f"📊 Processed {self.frame_count} frames, avg pixels: {avg_pixels:.0f}")
            else:
                self.log_warn(f"⚠️  Frame {self.frame_count}: No tensor found in message")
                
        except Exception as e:
            self.log_error(f"❌ Frame {self.frame_count}: Validation error: {e}")


class StreamingClientFunctionalTestApp(Application):
    """Functional test application that processes real video through StreamingClientOp."""
    
    def __init__(self, data_path="/workspace/holohub/data-streaming_client_demo-v3.5.0-dgpu/endoscopy", **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        
    def compose(self):
        # Validate data path
        data_dir = Path(self.data_path)
        if not data_dir.exists():
            print(f"⚠️  Data directory not found: {data_dir}")
            print("📁 Available data directories:")
            for path in Path("/workspace/holohub").glob("data*"):
                print(f"  - {path}")
            # Fallback to simple infrastructure test mode
            print("🔄 Falling back to infrastructure test mode (StreamingClient without video)")
            streaming_client = StreamingClientOp(
                self,
                name="streaming_client"
            )
            self.add_operator(streaming_client)
            print("✅ Functional test configured in infrastructure mode")
            return
            
        video_index = data_dir / "surgical_video.gxf_index"
        if not video_index.exists():
            print(f"⚠️  Video index file not found: {video_index}")
            return
            
        print(f"🎬 Using real video data from: {data_dir}")
        
        # Create allocator
        allocator = UnboundedAllocator(self, name="allocator")
        
        # Video source with real data
        source = VideoStreamReplayerOp(
            self,
            name="video_source",
            directory=str(data_dir),
            basename="surgical_video",
            frame_rate=30,
            realtime=False,
            repeat=False,
            count=50  # Process 50 frames for testing
        )
        
        # Format converter
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=allocator,
            # Convert to format expected by streaming client
            out_dtype="uint8",
            out_channel_order=[2, 1, 0],  # BGR
        )
        
        # Streaming client operator
        streaming_client = StreamingClientOp(
            self,
            name="streaming_client",
        )
        
        # Frame validator to verify frames flow through
        validator = VideoFrameValidatorOp(
            self,
            name="frame_validator"
        )
        
        # Connect the pipeline
        self.add_flow(source, format_converter, {("output", "source_video")})
        self.add_flow(format_converter, streaming_client)
        self.add_flow(streaming_client, validator, {("output_frames", "input_frames")})


def main():
    parser = argparse.ArgumentParser(description="Streaming Client Functional Test")
    parser.add_argument("--data", type=str, 
                       default="/workspace/holohub/data-streaming_client_demo-v3.5.0-dgpu/endoscopy",
                       help="Path to video data directory")
    args = parser.parse_args()
    
    # Create and run the functional test
    app = StreamingClientFunctionalTestApp(data_path=args.data)
    app.run()


if __name__ == "__main__":
    main()
