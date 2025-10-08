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

from operators.webrtc_client.webrtc_client_op import WebRTCClientOp

ROOT = os.path.dirname(__file__)


class WebAppThread(Thread):
    def __init__(self, webrtc_client_op, host, port, cert_file=None, key_file=None):
        super().__init__()
        self._webrtc_client_op = webrtc_client_op
        self._host = host
        self._port = port

        if cert_file:
            self._ssl_context = ssl.SSLContext()
            self._ssl_context.load_cert_chain(cert_file, key_file)
        else:
            self._ssl_context = None

        app = web.Application()
        app.on_shutdown.append(self._on_shutdown)
        app.router.add_get("/", self._index)
        app.router.add_get("/client.js", self._javascript)
        app.router.add_post("/offer", self._offer)

        self._runner = web.AppRunner(app)

    async def _on_shutdown(self, app):
        self._webrtc_client_op.shutdown()

    async def _index(self, request):
        content = open(os.path.join(ROOT, "index.html"), "r").read()
        return web.Response(content_type="text/html", text=content)

    async def _javascript(self, request):
        content = open(os.path.join(ROOT, "client.js"), "r").read()
        return web.Response(content_type="application/javascript", text=content)

    async def _offer(self, request):
        params = await request.json()

        (sdp, type) = await self._webrtc_client_op.offer(params["sdp"], params["type"])

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


class WebRTCClientApp(holoscan.core.Application):
    def __init__(self, cmdline_args):
        super().__init__()
        self._cmdline_args = cmdline_args

    def compose(self):
        webrtc_client = WebRTCClientOp(self, name="WebRTC Client")
        video_sink = holoscan.operators.HolovizOp(
            self,
            name="Video Sink",
            window_title="WebRTC Client",
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

        self.add_flow(webrtc_client, video_sink, {("output", "receivers")})

        # start the web server in the background, this will call the WebRTC server operator
        # 'offer' method when a connection is established
        self._web_app_thread = WebAppThread(
            webrtc_client,
            self._cmdline_args.host,
            self._cmdline_args.port,
            self._cmdline_args.cert_file,
            self._cmdline_args.key_file,
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
    cmdline_args = parser.parse_args()

    if cmdline_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = WebRTCClientApp(cmdline_args)
    app.run()


if __name__ == "__main__":
    main()
