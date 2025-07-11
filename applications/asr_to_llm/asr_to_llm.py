# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import sys

import riva.client
from holoscan.conditions import BooleanCondition
from holoscan.core import Application
from holoscan.logger import LogLevel, set_log_level
from llm import LLMOp
from riva.client.argparse_utils import (
    add_asr_config_argparse_parameters,
    add_connection_argparse_parameters,
)
from riva.client.audio_io import get_default_input_device_info, list_input_devices
from riva_asr import RivaASROp
from streaming_print import StreamingPrintOp

sys.stderr = sys.stdout


def parse_args() -> argparse.Namespace:
    default_device_info = get_default_input_device_info()
    default_device_index = None if default_device_info is None else default_device_info["index"]
    parser = argparse.ArgumentParser(
        description="Streaming transcription from microphone via Riva AI Services",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=default_device_index,
        help="An input audio device to use.",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List input audio device indices."
    )
    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
    parser = add_connection_argparse_parameters(parser)
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        help="A number of frames per second in audio streamed from a microphone.",
        default=16000,
    )
    parser.add_argument(
        "--file-streaming-chunk",
        type=int,
        default=1600,
        help="A maximum number of frames in a audio chunk sent to server.",
    )
    args = parser.parse_args()
    return args


class STT_To_LLM(Application):
    """Speech-to-text transcription and Large Language Model application.

    Transcribes an audio file to text, uses an LLM to perform requests specified in yaml file.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def compose(self):
        llm = LLMOp(self, name="llm", **self.kwargs("LLMOp"))
        stop_execution_condition = BooleanCondition(self, name="stop_execution_condition")
        auth = riva.client.Auth(
            self.args.ssl_cert, self.args.use_ssl, self.args.server, self.args.metadata
        )
        riva_streaming = RivaASROp(self, auth, self.args, stop_execution_condition, name="riva")
        printer = StreamingPrintOp(self, name="printer")

        self.add_flow(
            riva_streaming,
            printer,
            {("is_complete", "is_complete"), ("riva_response", "riva_response")},
        )
        self.add_flow(printer, llm)


def main():
    set_log_level(LogLevel.WARN)
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return

    app = STT_To_LLM(args=args)
    config_file = os.path.join(os.path.dirname(__file__), "asr_to_llm.yaml")
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    main()
