# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

        app = web.Application()
        app.on_shutdown.append(self._on_shutdown)
        app.router.add_get("/", self._index)
        app.router.add_get("/client.js", self._javascript)
        app.router.add_post("/offer", self._offer)
        app.router.add_get("/iceServers", self._get_ice_servers)

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

        (sdp, type) = await self._webrtc_server_op.offer(params["sdp"], params["type"])

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": sdp, "type": type}),
        )

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, self._host, self._port, ssl_context=self._ssl_context)
        logging.info(f"Starting web server at {self._host}:{self._port}")
        loop.run_until_complete(site.start())
        loop.run_forever()


class WebRTCServerApp(holoscan.core.Application):
    def __init__(self, cmdline_args):
        super().__init__()
        self._cmdline_args = cmdline_args

    def compose(self):
        data = os.environ.get("HOLOHUB_DATA_PATH", "../data")
        video_replayer = holoscan.operators.VideoStreamReplayerOp(
            self,
            name="video_replayer",
            directory=os.path.join(data, "racerx"),
            basename="racerx",
            realtime=False,
            repeat=True,
        )
        # convert VideoFrame to Tensor, there is currently no support for VideoFrame in Holoscan Python
        video_source = holoscan.operators.FormatConverterOp(
            self,
            name="convert_video_to_tensor",
            out_dtype="rgb888",
            pool=holoscan.resources.UnboundedAllocator(self, name="pool"),
            cuda_stream_pool=holoscan.resources.CudaStreamPool(
                self,
                name="cuda_stream_pool",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
            ),
        )
        self.add_flow(video_replayer, video_source)

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
