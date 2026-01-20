# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import asyncio
import json
import logging
import os
import random
import ssl
from threading import Thread

import holoscan
import numpy as np
from aiohttp import web

from operators.webrtc_server.webrtc_server_op import WebRTCServerOp

ROOT = os.path.dirname(__file__)


def parse_ice_strings(ice_server_strings):
    """
    Convert a list of ICE server strings into a list of dictionaries of the iceServers format.
    The iceServers format is from https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/RTCPeerConnection#iceservers

    Parameters:
    - ice_server_strings (list): A list of ICE server strings. Each string is expected to be
      in the format 'protocol:host:port[username:credential]' where '[username:credential]' is needed for TURN server.


    Returns:
    list: A list of dictionaries, where each dictionary represents an ICE server.
          Each dictionary has at least a 'urls' key, and may include 'username' and 'credential' keys.

    Example:
    If input_list is ['turn:10.0.0.131:3478[admin:admin]', 'stun:stun.l.google.com:19302'],
    the output will be [
        {
            'urls': 'turn:10.0.0.131:3478',
            'username': 'admin',
            'credential': 'admin',
        },
        {
            'urls': 'stun:stun.l.google.com:19302',
        },
    ]
    """
    ice_server_list = []
    for ice_string in ice_server_strings:
        if "[" in ice_string:
            parts = ice_string.split("[")
            url = parts[0]
            creds = parts[1].split(":")
            if len(creds) != 2:
                raise ValueError(
                    "Expected ice-server with credentials to be in form of turn:<ip>:<port>[<username>:<password>] "
                )
            username = creds[0]
            password = creds[1][:-1]
            ice_server = {"urls": url, "username": username, "credential": password}
        else:
            ice_server = {"urls": ice_string}
        ice_server_list.append(ice_server)
    return ice_server_list


class WebAppThread(Thread):
    def __init__(
        self,
        geometry_generator_op,
        webrtc_server_op,
        host,
        port,
        cert_file=None,
        key_file=None,
        ice_server=None,
    ):
        super().__init__()
        self._geometry_generator_op = geometry_generator_op
        self._webrtc_server_op = webrtc_server_op
        self._host = host
        self._port = port

        if cert_file:
            self._ssl_context = ssl.SSLContext()
            self._ssl_context.load_cert_chain(cert_file, key_file)
        else:
            self._ssl_context = None

        self._ice_servers = []
        if ice_server:
            self._ice_servers = parse_ice_strings(ice_server)

        app = web.Application()
        app.on_shutdown.append(self._on_shutdown)
        app.router.add_get("/", self._index)
        app.router.add_get("/client.js", self._javascript)
        app.router.add_post("/offer", self._offer)
        app.router.add_get("/iceServers", self._get_ice_servers)
        app.router.add_post("/config", self._config)

        self._runner = web.AppRunner(app)

    async def _on_shutdown(self, app):
        self._webrtc_server_op.shutdown()

    async def _index(self, request):
        content = open(os.path.join(ROOT, "index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)

    async def _javascript(self, request):
        content = open(os.path.join(ROOT, "client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

    async def _get_ice_servers(self, request):
        logging.info(f"Available ice servers are {json.dumps(self._ice_servers)}")
        return web.Response(content_type="application/json", text=json.dumps(self._ice_servers))

    async def _offer(self, request):
        params = await request.json()

        sdp, type = await self._webrtc_server_op.offer(params["sdp"], params["type"])

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": sdp, "type": type}),
        )

    async def _config(self, request):
        params = await request.json()

        self._geometry_generator_op.config(params)

        return web.Response()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, self._host, self._port, ssl_context=self._ssl_context)
        logging.info(f"Starting web server at {self._host}:{self._port}")
        loop.run_until_complete(site.start())
        loop.run_forever()


# Define custom Operators for use in the demo
class GeometryGenerationOp(holoscan.core.Operator):
    """Example creating geometric primitives."""

    def __init__(self, fragment, *args, **kwargs):
        self._text = "Text"
        self._speed = random.uniform(0.05, 0.15)
        self._x_pos = random.uniform(0.0, 1.0)
        self._y_pos = random.uniform(0.0, 1.0)
        self._x_dir = random.uniform(0.0, 0.1)
        self._y_dir = random.uniform(0.0, 0.1)
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.output("outputs")
        spec.output("output_specs")

    def config(self, config):
        self._text = config["text"]
        self._speed = float(config["speed"]) / 1000.0

    def compute(self, op_input, op_output, context):
        ####################################
        # Create a tensor for "dynamic_text"
        ####################################
        # Set of two (x, y) points marking the location of text
        dynamic_text = np.asarray(
            [
                (self._x_pos, self._y_pos),
            ],
            dtype=np.float32,
        )
        dynamic_text = dynamic_text[np.newaxis, :, :]

        out_message = {
            "dynamic_text": dynamic_text,
        }

        self._x_pos += self._x_dir * self._speed
        if self._x_pos < 0.0 or self._x_pos > 1.0:
            self._x_dir = -self._x_dir
        self._y_pos += self._y_dir * self._speed
        if self._y_pos < 0.0 or self._y_pos > 1.0:
            self._y_dir = -self._y_dir

        # emit the tensors
        op_output.emit(out_message, "outputs")

        ########################################
        # Create a input spec for "dynamic_text"
        ########################################
        # To dynamically change the input spec create a list of HolovizOp.InputSpec objects
        # and pass it to Holoviz.
        # All properties of the input spec (type, color, text, line width, ...) can be changed
        # dynamically.
        specs = []
        spec = holoscan.operators.HolovizOp.InputSpec("dynamic_text", "text")
        spec.text = [self._text]
        spec.color = [
            random.uniform(0.0, 1.0),
            random.uniform(0.0, 1.0),
            random.uniform(0.0, 1.0),
            1.0,
        ]
        specs.append(spec)
        # emit the output specs
        op_output.emit(specs, "output_specs")


class WebRTCServerApp(holoscan.core.Application):
    def __init__(self, cmdline_args):
        super().__init__()
        self._cmdline_args = cmdline_args

    def compose(self):
        cuda_stream_pool = holoscan.resources.CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        geometry_generator = GeometryGenerationOp(
            self,
            name="Geometry generator",
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="Visualizer",
            window_title="WebRTC Video Source",
            cuda_stream_pool=cuda_stream_pool,
            enable_render_buffer_output=True,
            headless=True,
            allocator=holoscan.resources.UnboundedAllocator(self, name="Holoviz allocator"),
        )
        # convert VideoFrame to Tensor, there is currently no support for VideoFrame in Holoscan Python
        format_converter = holoscan.operators.FormatConverterOp(
            self,
            name="convert_video_to_tensor",
            in_dtype="rgba8888",
            out_dtype="rgb888",
            cuda_stream_pool=cuda_stream_pool,
            pool=holoscan.resources.UnboundedAllocator(self, name="FormatConverter allocator"),
        )

        self.add_flow(geometry_generator, visualizer, {("outputs", "receivers")})
        self.add_flow(geometry_generator, visualizer, {("output_specs", "input_specs")})
        self.add_flow(visualizer, format_converter, {("render_buffer_output", "source_video")})

        # create the WebRTC server operator
        webrtc_server = WebRTCServerOp(self, name="webrtc_server")
        self.add_flow(format_converter, webrtc_server)

        # start the web server in the background, this will call the WebRTC server operator
        # 'offer' method when a connection is established
        self._web_app_thread = WebAppThread(
            geometry_generator,
            webrtc_server,
            self._cmdline_args.host,
            self._cmdline_args.port,
            self._cmdline_args.cert_file,
            self._cmdline_args.key_file,
            self._cmdline_args.ice_server,
        )
        self._web_app_thread.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--ice-server",
        action="append",
        help="ICE server config in the form of `turn:<ip>:<port>[<username>:<password>]` or `stun:<ip>:<port>`. "
        "This option can be specified multiple times to add multiple ICE servers.",
    )
    cmdline_args = parser.parse_args()

    if cmdline_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = WebRTCServerApp(cmdline_args)
    app.run()


if __name__ == "__main__":
    main()
