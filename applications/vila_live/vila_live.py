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
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from PIL import Image
from vlm import VLM
from webserver import Webserver


class VLMWebAppOp(Operator):
    """
    VLMWebApp that using a local VLM model and a Flask web-app to display the results
    """

    def __init__(self, fragment, *args, **kwargs):
        self.server = Webserver()
        self.vlm = VLM()
        self.is_busy = Event()
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Start the Webserver on a background thread
        self.server.start()
        time.sleep(3)

    def setup(self, spec: OperatorSpec):
        spec.input("video_stream")

    def stop(self):
        pass

    def annotate_image(self, image_b64):
        self.is_busy.set()
        prompt = self.server.user_input.replace('"', "")
        for response in self.vlm.generate_response(prompt, image_b64):
            chat_history = [[prompt, response]]
            self.server.send_chat_history(chat_history)
        # time.sleep(3) # May be needed depending on the speed of the model
        self.is_busy.clear()

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("video_stream").get("")
        if in_message:
            # Create a b64 Image from the Holoscan Tensor
            cp_image = cp.from_dlpack(in_message)
            np_image = cp.asnumpy(cp_image)
            image = Image.fromarray(np_image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")  # Save in JPEG format
            buffer.seek(0)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Check if we're currently running the VLM
            if not self.is_busy.is_set():
                thread = Thread(target=self.annotate_image, args=(image_b64,))
                thread.start()

            # Send the video frame to the web-app to be displayed
            payload = {"image_b64": image_b64}
            self.server.send_message(payload)


class V4L2toVLM(Application):
    def __init__(self):
        """V4L2 to VLM app"""
        super().__init__()
        # set name
        self.name = "V4L2 to VLM app"

    def compose(self):
        # V4L2 to capture usb camera input
        source = V4L2VideoCaptureOp(
            self,
            name="source",
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("source"),
        )

        formatter_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        format_converter_vlm = FormatConverterOp(
            self,
            name="convert_video_to_tensor",
            in_dtype="rgba8888",
            out_dtype="rgb888",
            cuda_stream_pool=formatter_cuda_stream_pool,
            pool=UnboundedAllocator(self, name="FormatConverter allocator"),
        )

        holoviz_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            window_title="VILA Live",
            headless=True,
            enable_render_buffer_input=False,
            enable_render_buffer_output=True,
            allocator=UnboundedAllocator(self, name="Holoviz allocator"),
            cuda_stream_pool=holoviz_cuda_stream_pool,
            **self.kwargs("holoviz"),
        )

        # Initialize the VLM + WebApp operator
        web_server = VLMWebAppOp(self, name="VLMWebAppOp")

        self.add_flow(source, visualizer, {("signal", "receivers")})
        self.add_flow(visualizer, format_converter_vlm, {("render_buffer_output", "source_video")})
        self.add_flow(format_converter_vlm, web_server, {("tensor", "video_stream")})


def main():
    # Parse args
    parser = ArgumentParser(description="VILA live application.")
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "vila_live.yaml")
    else:
        config_file = args.config

    app = V4L2toVLM()
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
