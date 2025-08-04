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

from holoscan.operators import FormatConverterOp, V4L2VideoCaptureOp, VideoStreamReplayerOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator

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
        webrtc_server_op,
        host,
        port,
        cert_file=None,
        key_file=None,
        ice_server=None,
        serve_client=True,
    ):
        super().__init__()
        self._webrtc_server_op = webrtc_server_op
        self._host = host
        self._port = port
        self._serve_client = serve_client

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
            self._ice_servers = [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"},
            ]
            logging.info("No ICE servers configured, using default STUN servers")

        app = web.Application()
        app.on_shutdown.append(self._on_shutdown)

        # Add routes based on mode
        offer_resource = app.router.add_post("/offer", self._offer)
        ice_servers_resource = app.router.add_get("/iceServers", self._get_ice_servers)

        if self._serve_client:
            # Add client serving routes
            index_resource = app.router.add_get("/", self._index)
            js_resource = app.router.add_get("/client.js", self._javascript)
        else:
            # API-only mode: register OPTIONS handlers for CORS preflight
            if not CORS_AVAILABLE:
                app.router.add_options("/offer", self._options_handler)
                app.router.add_options("/iceServers", self._options_handler)

        # Add CORS support (for both modes - doesn't hurt same-origin clients)
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
            # Add CORS to API routes
            cors.add(offer_resource)
            cors.add(ice_servers_resource)
            if self._serve_client:
                cors.add(index_resource)
                cors.add(js_resource)

        self._runner = web.AppRunner(app)

    async def _on_shutdown(self, app):
        await self._webrtc_server_op.shutdown()

    async def _index(self, request):
        content = open(os.path.join(ROOT, "index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)

    async def _javascript(self, request):
        content = open(os.path.join(ROOT, "client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

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
        try:
            params = await request.json()
            if "sdp" not in params or "type" not in params:
                return web.Response(status=400, text="Missing 'sdp' or 'type' in request")
            (sdp, type) = await self._webrtc_server_op.offer(params["sdp"], params["type"])
        except (json.JSONDecodeError, ValueError) as e:
            return web.Response(status=400, text=f"Invalid request: {str(e)}")

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
        mode = "with embedded client" if self._serve_client else "API-only mode"
        logging.info(f"Starting WebRTC server at {self._host}:{self._port} ({mode}, {cors_status})")
        if self._serve_client:
            logging.info(f"  Web UI: http://{self._host}:{self._port}/")
        logging.info("  API endpoints:")
        logging.info(f"    POST http://{self._host}:{self._port}/offer - WebRTC signaling")
        logging.info(
            f"    GET  http://{self._host}:{self._port}/iceServers - ICE server configuration"
        )

        loop.run_until_complete(site.start())
        loop.run_forever()


class WebRTCServerApp(holoscan.core.Application):
    def __init__(self, cmdline_args):
        super().__init__()
        self._cmdline_args = cmdline_args

    def compose(self):
        # Create video source based on selected mode
        if self._cmdline_args.source == "camera":
            logging.info(f"Using camera source: {self._cmdline_args.camera_device}")
            video_input = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=UnboundedAllocator(self, name="input_pool"),
                device=self._cmdline_args.camera_device,
                pixel_format=self._cmdline_args.pixel_format,
            )
            # Camera outputs based on configured pixel format
            format_converter = FormatConverterOp(
                self,
                name="convert_video_to_tensor",
                in_dtype=self._cmdline_args.pixel_format.lower(),
                out_dtype="rgb888",
                pool=UnboundedAllocator(self, name="converter_pool"),
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
        else:  # file mode
            video_file = self._cmdline_args.video_file
            if not video_file:
                # Use default video file
                data = os.environ.get("HOLOHUB_DATA_PATH", "../data")
                video_dir = os.path.join(data, "racerx")
                video_file = "racerx"
                logging.info(f"Using default video file: {video_dir}/{video_file}")
            else:
                # User provided video file path
                video_dir = os.path.dirname(video_file) or "."
                video_file = os.path.splitext(os.path.basename(video_file))[0]
                logging.info(f"Using video file: {video_dir}/{video_file}")

            video_input = VideoStreamReplayerOp(
                self,
                name="video_replayer",
                directory=video_dir,
                basename=video_file,
                realtime=False,
                repeat=True,
            )
            # Video replayer outputs VideoFrame, convert to Tensor
            format_converter = FormatConverterOp(
                self,
                name="convert_video_to_tensor",
                out_dtype="rgb888",
                pool=UnboundedAllocator(self, name="converter_pool"),
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

        # Connect the pipeline
        self.add_flow(video_input, format_converter)

        # create the WebRTC server operator
        webrtc_server = WebRTCServerOp(self, name="webrtc_server")
        self.add_flow(format_converter, webrtc_server)

        # start the web server in the background, this will call the WebRTC server operator
        # 'offer' method when a connection is established
        self._web_app_thread = WebAppThread(
            webrtc_server,
            self._cmdline_args.host,
            self._cmdline_args.port,
            self._cmdline_args.cert_file,
            self._cmdline_args.key_file,
            self._cmdline_args.ice_server,
            self._cmdline_args.serve_client,
        )
        self._web_app_thread.start()


def main():
    parser = argparse.ArgumentParser(
        description="WebRTC Video Streaming Server - Supports both file replay and live camera streaming"
    )

    # Video source options
    source_group = parser.add_argument_group("Video Source Options")
    source_group.add_argument(
        "--source",
        choices=["file", "camera"],
        default="file",
        help="Video source type: 'file' for video file replay or 'camera' for live camera capture (default: file)",
    )
    source_group.add_argument(
        "--video-file",
        help="Path to video file (for source=file). If not specified, uses default racerx video. "
        "Provide full path or just basename (e.g., /path/to/video.gxf or video)",
    )
    source_group.add_argument(
        "--camera-device",
        default="/dev/video0",
        help="Camera device path (for source=camera, default: /dev/video0)",
    )
    source_group.add_argument(
        "--pixel-format",
        default="YUYV",
        help="Camera pixel format (for source=camera, default: YUYV)",
    )

    # Server options
    server_group = parser.add_argument_group("Server Options")
    server_group.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    server_group.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    server_group.add_argument(
        "--serve-client",
        action="store_true",
        default=True,
        help="Serve embedded web client UI (default: True)",
    )
    server_group.add_argument(
        "--no-serve-client",
        dest="serve_client",
        action="store_false",
        help="API-only mode - do not serve embedded client (enables full CORS for external clients)",
    )
    server_group.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    server_group.add_argument("--key-file", help="SSL key file (for HTTPS)")

    # WebRTC options
    webrtc_group = parser.add_argument_group("WebRTC Options")
    webrtc_group.add_argument(
        "--ice-server",
        action="append",
        help="ICE server config in the form of `turn:<ip>:<port>[<username>:<password>]` or `stun:<ip>:<port>`. "
        "This option can be specified multiple times to add multiple ICE servers. "
        "If not specified, default STUN servers will be used.",
    )

    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    cmdline_args = parser.parse_args()

    if cmdline_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Log configuration
    logging.info("=" * 60)
    logging.info("WebRTC Video Streaming Server")
    logging.info("=" * 60)
    logging.info(f"Video Source: {cmdline_args.source}")
    if cmdline_args.source == "camera":
        logging.info(f"Camera Device: {cmdline_args.camera_device}")
        logging.info(f"Pixel Format: {cmdline_args.pixel_format}")
    else:
        logging.info(f"Video File: {cmdline_args.video_file or 'default (racerx)'}")
    logging.info(f"Server Mode: {'Embedded Client' if cmdline_args.serve_client else 'API-Only'}")
    logging.info("=" * 60)

    app = WebRTCServerApp(cmdline_args)
    app.run()


if __name__ == "__main__":
    main()
