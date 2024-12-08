# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
import io
import os
import time
from argparse import ArgumentParser
from threading import Event, Thread

import cupy as cp
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp, VideoStreamReplayerOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from PIL import Image, ImageDraw, ImageFont
from vlm import VLM
from webserver import Webserver
from datetime import datetime, timedelta


class VLMWebAppOp(Operator):
    """
    VLMWebApp that uses a local VLM model and a Flask web-app to display the results
    """

    def __init__(self, fragment, *args, **kwargs):
        self.server = Webserver()
        self.vlm = VLM()
        self.is_busy = Event()
        self.frame_count = 0  # Initialize frame counter
        self.start_time = datetime.now()  # Store the start time
        self.frame_rate = 12
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Start the Webserver on a background thread
        self.server.start()
        time.sleep(3)

    def setup(self, spec: OperatorSpec):
        # Support multiple video streams
        spec.input("video_stream1")
        spec.input("video_stream2")
        spec.input("video_stream3")

    def stop(self):
        pass

    def annotate_image(self, image_b64, frame_number, timestamp, source):
        self.is_busy.set()
        prompt = self.server.user_input.replace('"', "")
        
        full_response = ""
        current_frame = frame_number
        
        for response in self.vlm.generate_response(prompt, image_b64):
            if frame_number != current_frame:
                # Frame has changed, send the previous complete message
                if full_response:
                    response_with_frame_number_timestamp = f"[Frame: {current_frame}, Timestamp: {timestamp}, Source: {source}] {full_response}"
                    prompt_with_Q = f"Prompt: {prompt}"
                    chat_history = [[prompt_with_Q, response_with_frame_number_timestamp]]
                    self.server.send_chat_history(chat_history)
                
                # Reset for the new frame
                full_response = ""
                current_frame = frame_number
            
            full_response = response  # Update with the latest complete response

        # Send the final response for the last frame
        if full_response:
            response_with_frame_number_timestamp = f"[Frame: {current_frame}, Timestamp: {timestamp}, Source: {source}] {full_response}"
            prompt_with_Q = f"Prompt: {prompt}"
            chat_history = [[prompt_with_Q, response_with_frame_number_timestamp]]
            self.server.send_chat_history(chat_history)

        self.is_busy.clear()

    def compute(self, op_input, op_output, context):
        # Attempt to receive from both input streams
        in_message1 = op_input.receive("video_stream1").get("")
        in_message2 = op_input.receive("video_stream2").get("")
        in_message3 = op_input.receive("video_stream3").get("")

        # Process first video stream
        if in_message1:
            self._process_video_stream(in_message1, "source1")

        # Process second video stream
        if in_message2:
            self._process_video_stream(in_message2, "source2")

        if in_message3:
            self._process_video_stream(in_message3, "source3")

    def _process_video_stream(self, in_message, source):
        # Increment the frame count
        self.frame_count += 1
        
        # Create a b64 Image from the Holoscan Tensor
        cp_image = cp.from_dlpack(in_message)
        np_image = cp.asnumpy(cp_image)
        image = Image.fromarray(np_image)

        # Draw the frame number and source on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_position = (10, 10)  # Position at the top-left corner
        text_color = (255, 100, 255)  # Magenta text color
        draw.text(text_position, f"{source} - Frame: {self.frame_count}", fill=text_color, font=font)

        # Save the image for verification
        image.save(f'test_image_{source}.jpg')

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Check if we're currently running the VLM
        if not self.is_busy.is_set():
            hours, minutes, seconds = get_timestamp_in_hms(self.frame_count, self.frame_rate)
            timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
            thread = Thread(target=self.annotate_image, args=(image_b64, self.frame_count, timestamp, source))
            thread.start()

        # Send the video frame to the web-app to be displayed
        payload = {"image_b64": image_b64, "source": source}
        self.server.send_message(payload)


class V4L2toVLM(Application):
    def __init__(self, data, source="v4l2", video_device="none"):
        """V4L2 to VLM app"""
        super().__init__()
        # set name
        self.name = "V4L2 to VLM app"
        self.source = source

        if data == "none":
            data = "/workspace/holohub/data/vila_live"

        self.sample_data_path = data
        self.video_device = video_device

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        # Sources initialization
        sources = []
        source_outputs = []

        # Try sources with sequential numbering
        for i in range(3):  # Start with trying v4l2_source_0 and v4l2_source_1
            source_key = f'v4l2_source_{i}'
            v4l2_args = self.kwargs(source_key)

            # Skip if no configuration found for this source
            if not v4l2_args:
                continue

            # Override device if video_device is specified
            if self.video_device != "none":
                v4l2_args["device"] = self.video_device
                

            # Create source
            source = V4L2VideoCaptureOp(
                self,
                name=f"v4l2_source{i+1}",
                allocator=pool,
                **v4l2_args,
            )

            sources.append(source)
            source_outputs.append("signal")

        # If no sources found, fall back to default
        if not sources:
            source = V4L2VideoCaptureOp(
                self,
                name="v4l2_source1",
                allocator=pool,
                device="/dev/video0"
            )
            sources.append(source)
            source_outputs.append("signal")

        # Cuda Stream Pools
        formatter_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # Format converters for both sources
        format_converters = []
        for i in range(len(sources)):
            format_converter = FormatConverterOp(
                self,
                name=f"convert_video_to_tensor{i+1}",
                in_dtype="rgba8888",
                out_dtype="rgb888",
                cuda_stream_pool=formatter_cuda_stream_pool,
                pool=UnboundedAllocator(self, name=f"FormatConverter allocator{i+1}"),
            )
            format_converters.append(format_converter)

        # Holoviz for visualization
        holoviz_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        visualizers = []
        for i in range(len(sources)):
            visualizer = HolovizOp(
                self,
                name=f"holoviz{i+1}",
                window_title=f"VILA Live Source {i+1}",
                headless=True,
                enable_render_buffer_input=False,
                enable_render_buffer_output=True,
                allocator=UnboundedAllocator(self, name=f"Holoviz allocator{i+1}"),
                cuda_stream_pool=holoviz_cuda_stream_pool,
                **self.kwargs("holoviz"),
            )
            visualizers.append(visualizer)

        # Initialize the VLM + WebApp operator
        web_server = VLMWebAppOp(self, name="VLMWebAppOp")

        # Create flows for each source
        for i, (source, visualizer, format_converter) in enumerate(zip(sources, visualizers, format_converters)):
            print(f"value of i:{i}")
            self.add_flow(source, visualizer, {(source_outputs[i], "receivers")})
            self.add_flow(visualizer, format_converter, {("render_buffer_output", "source_video")})
            self.add_flow(format_converter, web_server, {("tensor", f"video_stream{i+1}")})


def get_timestamp_in_hms(frame_count, frame_rate):
    total_seconds = frame_count / frame_rate
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return int(hours), int(minutes), int(seconds)


def main():
    # Parse args
    parser = ArgumentParser(description="VILA live application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help=(
            "If 'v4l2', uses the v4l2 device specified in the yaml file or "
            " --video_device if specified. "
            "If 'replayer', uses video stream replayer."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    parser.add_argument(
        "-v",
        "--video_device",
        default="none",
        help=("The video device to use.  By default the application will use /dev/video0"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "vila_live.yaml")
    else:
        config_file = args.config

    app = V4L2toVLM(args.data, args.source, args.video_device)
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
