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

import argparse
import logging
import os
import sys

import riva.client
from agent_framework_op import AgentFrameworkOp
from holoscan.core import Application
from holoscan.logger import LogLevel, set_log_level
from riva.client import audio_io
from riva.client.argparse_utils import (
    add_asr_config_argparse_parameters,
    add_connection_argparse_parameters,
)
from riva_asr import RivaASROp
from riva_tts import RivaTTSOp
from streaming_print import StreamingPrintOp
from webserver import WebServerOp

sys.stderr = sys.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming transcription from microphone via Riva AI Services",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )
    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
    parser = add_connection_argparse_parameters(parser)
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        help="A number of frames per second in audio streamed from a microphone.",
        default=None,
    )
    parser.add_argument(
        "--voice",
        help="A voice name to use. If this parameter is missing, then the server will try a first available model "
        "based on parameter `--language-code`.",
    )
    # Overwrite Riva client '--ssl-cert' arg resulting from riva.client.argparse_utils
    parser.add_argument(
        "--ssl-cert",
        type=str,
        help="Absolute path to 'cert.pem' inside container file system",
        default="/workspace/holohub/applications/ehr_query_llm/lmm/ssl/cert.pem",
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        help="Absolute path to 'key.pem' inside container file system",
        default="/workspace/holohub/applications/ehr_query_llm/lmm/ssl/key.pem",
    )
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument(
        "--log-level",
        choices=log_levels,
        type=str,
        help="Log level to control granularity of output",
        default="DEBUG",
    )
    args = parser.parse_args()
    return args


class ASR_LLM_TTS(Application):
    """Use Riva ASR to transcribe speech from the selected microphone, which is sent to an
    llm chatbot, whose response is then converted to speech using Riva TTS. This app includes a
    flask server to display a Chat interface.
    """

    def __init__(self, args):
        self.args = args
        # Set application's log level
        logging.basicConfig(level=args.log_level, format="%(levelname)s - %(name)s: %(message)s")
        super().__init__()

    def compose(self):
        agent = AgentFrameworkOp(self, name="agent", **self.kwargs("AgentFrameworkOp"))
        # Define auth token for Riva
        auth = riva.client.Auth(
            ssl_cert=None, use_ssl=False, uri=self.args.server, metadata_args=self.args.metadata
        )

        riva_asr = RivaASROp(self, auth, self.args, name="riva", **self.kwargs("RivaASROp"))
        printer = StreamingPrintOp(self, name="printer", **self.kwargs("StreamingPrintOp"))
        riva_tts = RivaTTSOp(self, auth, self.args, name="riva_tts", **self.kwargs("RivaTTSOp"))

        # Initialize the web app with relevant callbacks for UI buttons
        web_server = WebServerOp(
            self,
            audio_msg_callback=riva_asr.receive_audio,
            history_reset_callback=agent.reset_history,
            voice_change_callback=riva_tts.set_voice,
            ssl_cert=self.args.ssl_cert,
            ssl_key=self.args.ssl_key,
            webapp_port=8050,
            ws_port=49000,
            name="server",
        )

        # Redirects the tts audio to the web app for processing
        riva_tts.set_audio_callback(callback=web_server.server.send_message)

        self.add_flow(riva_asr, printer)
        self.add_flow(printer, agent)
        self.add_flow(agent, web_server, {("chat_history", "chat_history")})
        self.add_flow(agent, riva_tts, {("agent_response", "agent_response")})

        # Limits the noisy output from the webserver when debugging
        logging.getLogger("websockets.server").setLevel(logging.ERROR)

        # Limits the noisy output from the webserver when debugging
        logging.getLogger("websockets.server").setLevel(logging.ERROR)


def main():
    set_log_level(LogLevel.WARN)
    args = parse_args()
    if not args.sample_rate_hz:
        args.sample_rate_hz = int(audio_io.get_default_input_device_info()["defaultSampleRate"])

    app = ASR_LLM_TTS(args=args)
    config_file = os.path.join(os.path.dirname(__file__), "asr_llm_tts.yaml")
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
