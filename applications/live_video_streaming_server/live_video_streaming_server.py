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

import argparse
import asyncio
import json
import logging
import os
import ssl
from threading import Thread

import holoscan
from aiohttp import web

# Try to import aiohttp-cors, fall back to manual CORS headers if not available
try:
    from aiohttp_cors import ResourceOptions
    from aiohttp_cors import setup as cors_setup

    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    logging.warning("aiohttp-cors not available, using manual CORS headers")

from holoscan.operators import FormatConverterOp, V4L2VideoCaptureOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator

from operators.webrtc_server.webrtc_server_op import WebRTCServerOp

ROOT = os.path.dirname(__file__)

# Camera configuration (normally set in YAML)
CAMERA_DEVICE = "/dev/video0"
CAMERA_PIXEL_FORMAT = "YUYV"


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
        self, webrtc_server_op, host, port, cert_file=None, key_file=None, ice_server=None
    ):
        super().__init__()
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
        else:
            # Add default STUN servers when none are configured
            # These are required for WebRTC to work through NAT/firewalls
            self._ice_servers = [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"},
            ]
            logging.info("No ICE servers configured, using default STUN servers")

        app = web.Application()
        app.on_shutdown.append(self._on_shutdown)

        # Server-only API endpoints - no client file serving
        app.router.add_post("/offer", self._offer)
        app.router.add_get("/iceServers", self._get_ice_servers)
        # Register manual preflight handlers only when aiohttp-cors is not used
        if not CORS_AVAILABLE:
            app.router.add_options("/offer", self._options_handler)
            app.router.add_options("/iceServers", self._options_handler)

        # Add CORS support for external web applications
        if CORS_AVAILABLE:
            cors = cors_setup(
                app,
                defaults={
                    "*": ResourceOptions(
                        allow_credentials=True,
                        expose_headers="*",
                        allow_headers="*",
                        allow_methods="*",
                    )
                },
            )

            # Add CORS to our routes
            cors.add(app.router._resources[-2])  # /offer route
            cors.add(app.router._resources[-1])  # /iceServers route

        self._runner = web.AppRunner(app)

    async def _on_shutdown(self, app):
        self._webrtc_server_op.shutdown()

    # Removed _index and _javascript methods - no longer serving client files

    async def _options_handler(self, request):
        """Handle CORS preflight OPTIONS requests"""
        response = web.Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
        return response

    async def _get_ice_servers(self, request):
        logging.info(f"Available ice servers are {json.dumps(self._ice_servers)}")
        response = web.Response(content_type="application/json", text=json.dumps(self._ice_servers))

        # Add CORS headers for cross-origin requests only when aiohttp-cors is not available
        if not CORS_AVAILABLE:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"

        return response

    async def _offer(self, request):
        params = await request.json()

        (sdp, type) = await self._webrtc_server_op.offer(params["sdp"], params["type"])

        response = web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": sdp, "type": type}),
        )

        # Add CORS headers for cross-origin requests only when aiohttp-cors is not available
        if not CORS_AVAILABLE:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"

        return response

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, self._host, self._port, ssl_context=self._ssl_context)

        cors_status = "with aiohttp-cors" if CORS_AVAILABLE else "with manual CORS headers"
        logging.info(
            f"Starting WebRTC streaming server (API only) at {self._host}:{self._port} ({cors_status})"
        )
        logging.info("Available endpoints:")
        logging.info(f"  POST http://{self._host}:{self._port}/offer - WebRTC signaling")
        logging.info(
            f"  GET  http://{self._host}:{self._port}/iceServers - ICE server configuration"
        )
        loop.run_until_complete(site.start())
        loop.run_forever()


class WebRTCServerApp(holoscan.core.Application):
    def __init__(self, cmdline_args):
        super().__init__()
        self._cmdline_args = cmdline_args
        # Camera parameters (could be set from YAML or elsewhere)
        self.device = CAMERA_DEVICE
        self.pixel_format = CAMERA_PIXEL_FORMAT

    def compose(self):
        # Use V4L2VideoCaptureOp instead of VideoStreamReplayerOp
        v4l2_source = V4L2VideoCaptureOp(
            self,
            name="v4l2_source",
            allocator=UnboundedAllocator(self, name="pool"),
            device=self.device,
            pixel_format=self.pixel_format,
            # Optionally add width, height, etc. if needed
        )
        video_source = FormatConverterOp(
            self,
            name="convert_video_to_tensor",
            in_dtype="rgba8888",  # V4L2 usually outputs RGBA8888
            out_dtype="rgb888",
            pool=UnboundedAllocator(self, name="pool"),
            cuda_stream_pool=CudaStreamPool(
                self,
                name="cuda_stream_pool",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
            ),
        )
        self.add_flow(v4l2_source, video_source)

        # create the WebRTC server operator
        webrtc_server = WebRTCServerOp(self, name="webrtc_server")
        self.add_flow(video_source, webrtc_server)

        # start the web server in the background, this will call the WebRTC server operator
        # 'offer' method when a connection is established
        self._web_app_thread = WebAppThread(
            webrtc_server,
            self._cmdline_args.host,
            self._cmdline_args.port,
            self._cmdline_args.cert_file,
            self._cmdline_args.key_file,
            self._cmdline_args.ice_server,
        )
        self._web_app_thread.start()


if __name__ == "__main__":
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
