# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# flask webserver

# Original Code: https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/llamaspeak/webserver.py

import copy
import json
import pprint
import queue
import ssl
import struct
import threading
import time

import flask
from holoscan.core import Operator, OperatorSpec
from websockets.sync.server import serve as websocket_serve


class Webserver(threading.Thread):
    """
    Flask + websockets server for the chat interface
    """

    def __init__(
        self,
        web_server="0.0.0.0",
        web_port=8050,
        ws_port=49000,
        ssl_cert=None,
        ssl_key=None,
        audio_msg_callback=None,
        voice_change_callback=None,
        history_reset_callback=None,
        log_level=0,
        **kwargs,
    ):
        super(Webserver, self).__init__(daemon=True)  # stop thread on main() exit

        self.host = web_server
        self.port = web_port

        self.log_level = log_level

        self.msg_count_rx = 0
        self.msg_count_tx = 0

        self.audio_msg_callback = audio_msg_callback
        self.voice_change_callback = voice_change_callback
        self.history_reset_callback = history_reset_callback

        # SSL / HTTPS
        self.ssl_key = ssl_key
        self.ssl_cert = ssl_cert
        self.ssl_context = None

        if self.ssl_cert and self.ssl_key:
            # self.ssl_context = (self.ssl_cert, self.ssl_key)
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile=self.ssl_cert, keyfile=self.ssl_key)

        # flask server
        self.app = flask.Flask(__name__)
        self.app.add_url_rule("/", view_func=self.on_index, methods=["GET"])

        # websocket
        self.ws_port = ws_port
        self.ws_queue = queue.Queue()

        self.ws_server = websocket_serve(
            self.on_websocket, host=self.host, port=self.ws_port, ssl_context=self.ssl_context
        )
        self.ws_thread = threading.Thread(
            target=lambda: self.ws_server.serve_forever(), daemon=True
        )

    def on_index(self):
        return flask.render_template("index.html", websocket_port=self.ws_port)

    def on_websocket_msg(self, msg, type, timestamp):
        """
        Receive websocket message from client
        """
        if type == 0:  # JSON
            if "chat_history_reset" in msg:
                self.history_reset_callback()
            if "tts_voice" in msg:
                self.voice_change_callback(msg["tts_voice"])
            pass
        elif type == 1:  # text (chat input)
            # TODO enable text input
            # self.on_llm_prompt(msg)
            pass
        elif type == 2:  # web audio (mic)
            self.audio_msg_callback(msg)

    def on_websocket(self, websocket):
        print(f"-- new websocket connection from {websocket.remote_address}")

        # empty the queue from before the connection was made
        # (otherwise client will be flooded with old messages)
        # TODO implement self.connected so the ws_queue doesn't grow so large without webclient connected...
        while True:
            try:
                self.ws_queue.get(block=False)
            except queue.Empty:
                break

        self.on_websocket_msg({"client_state": "connected"}, 0, int(time.time() * 1000))

        listener_thread = threading.Thread(
            target=self.websocket_listener, args=[websocket], daemon=True
        )
        listener_thread.start()

        while True:
            websocket.send(self.ws_queue.get())  # json.dumps(self.ws_queue.get()))

    def websocket_listener(self, websocket):
        print(f"-- listening on websocket connection from {websocket.remote_address}")

        header_size = 32

        while True:
            msg = websocket.recv()

            if isinstance(msg, str):
                print(
                    f'-- warning:  dropping text-mode websocket message from {websocket.remote_address} "{msg}"'
                )
                continue

            if len(msg) <= header_size:
                print(
                    f"-- warning:  dropping invalid websocket message from {websocket.remote_address} (size={len(msg)})"
                )
                continue

            msg_id, timestamp, magic_number, msg_type, payload_size, _, _ = struct.unpack_from(
                "!QQHHIII", msg
            )

            if magic_number != 42:
                print(
                    f"-- warning:  dropping invalid websocket message from {websocket.remote_address} (magic_number={magic_number} size={len(msg)})"
                )
                continue

            if msg_id != self.msg_count_rx:
                print(
                    f"-- warning:  Received websocket message from {websocket.remote_address} with out-of-order ID {msg_id}  (last={self.msg_count_rx})"
                )
                self.msg_count_rx = msg_id

            self.msg_count_rx += 1
            msgPayloadSize = len(msg) - header_size

            if payload_size != msgPayloadSize:
                print(
                    f"-- warning:  Received invalid websocket message from {websocket.remote_address} (payload_size={payload_size} actual={msgPayloadSize}"
                )

            payload = msg[header_size:]

            if msg_type == 0:  # json
                payload = json.loads(payload)
            elif msg_type == 1:  # text
                payload = payload.decode("utf-8")

            if self.log_level > 1 or (self.log_level > 0 and msg_type <= 1):
                print(
                    f"-- Received {Webserver.msg_type_str(msg_type)} websocket message from {websocket.remote_address} (type={msg_type} size={payload_size})"
                )
                if msg_type <= 1:
                    pprint.pprint(payload)

            self.on_websocket_msg(payload, msg_type, timestamp)

    def send_message(self, payload, type=None, timestamp=None):
        if timestamp is None:
            timestamp = time.time() * 1000

        encoding = None

        if type is None:
            if isinstance(payload, str):
                type = 1
                encoding = "utf-8"
            elif isinstance(payload, bytes):
                type = 2
            else:
                type = 0
                encoding = "ascii"

        if self.log_level > 1 or (self.log_level > 0 and type <= 1):
            print(
                f"-- sending {Webserver.msg_type_str(type)} websocket message (type={type} size={len(payload)})"
            )
            if type <= 1:
                pprint.pprint(payload)

        if type == 0 and not isinstance(
            payload, str
        ):  # json.dumps() might have already been called
            # print('sending JSON', payload)
            payload = json.dumps(payload)

        if not isinstance(payload, bytes):
            if encoding is not None:
                payload = bytes(payload, encoding=encoding)
            else:
                payload = bytes(payload)

        # do we even need this queue at all and can the websocket just send straight away?
        self.ws_queue.put(
            b"".join(
                [
                    #
                    # 32-byte message header format:
                    #
                    #   0   uint64  message_id    (message_count_tx)
                    #   8   uint64  timestamp     (milliseconds since Unix epoch)
                    #   16  uint16  magic_number  (42)
                    #   18  uint16  message_type  (0=json, 1=text, >=2 binary)
                    #   20  uint32  payload_size  (in bytes)
                    #   24  uint32  unused        (padding)
                    #   28  uint32  unused        (padding)
                    #
                    struct.pack(
                        "!QQHHIII",
                        self.msg_count_tx,
                        int(timestamp),
                        42,
                        type,
                        len(payload),
                        0,
                        0,
                    ),
                    payload,
                ]
            )
        )

        self.msg_count_tx += 1

    def send_chat_history(self, history):
        history = copy.deepcopy(history)

        def translate_web(text):
            text = text.replace("\n", "<br/>")
            return text

        for n in range(len(history)):
            for m in range(len(history[n])):
                history[n][m] = translate_web(history[n][m])

        # print("-- sending chat history", history)
        self.send_message({"chat_history": history})

    @staticmethod
    def msg_type_str(type):
        if type == 0:
            return "json"
        elif type == 1:
            return "text"
        else:
            return "binary"

    def run(self):
        print(f"-- starting webserver @ {self.host}:{self.port}")
        self.ws_thread.start()
        self.app.run(
            host=self.host,
            port=self.port,
            ssl_context=self.ssl_context,
            debug=True,
            use_reloader=False,
        )


class WebServerOp(Operator):
    """
    WebServerOp that wraps Dusty-NV's Webserver implementation that
    uses websockets to update a Flask app.
    """

    def __init__(
        self,
        fragment,
        audio_msg_callback,
        history_reset_callback,
        voice_change_callback,
        ssl_key,
        ssl_cert,
        webapp_port,
        ws_port,
        *args,
        **kwargs,
    ):
        self.server = Webserver(
            audio_msg_callback=audio_msg_callback,
            history_reset_callback=history_reset_callback,
            voice_change_callback=voice_change_callback,
            ssl_key=ssl_key,
            ssl_cert=ssl_cert,
            web_port=webapp_port,
            ws_port=ws_port,
        )
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Start the Webserver on a background thread
        self.server.start()
        time.sleep(3)

    def setup(self, spec: OperatorSpec):
        spec.input("chat_history")

    def stop(self):
        # Stop the mic streaming & the key listener
        pass

    def compute(self, op_input, op_output, context):
        history = op_input.receive("chat_history")
        # Update the chat history with the latest history
        self.server.send_chat_history(history)
