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

import os

from holoscan.core import Operator, OperatorSpec


class StreamingPrintOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.temp_transcription = ["======== Current Transcription ========\n\n", ""]
        self.transcription = ["======== Current Transcription ========\n\n", ""]
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # Indicates that the Riva transcription is complete
        spec.input("is_complete")
        # The streaming Riva response
        spec.input("riva_response")
        # The final output
        spec.output("final_text")

    def print_streaming(self, response):
        num_chars_printed = 0  # used in 'no' additional_info
        if not response or not response.results:
            return
        partial_transcript = ""
        for result in response.results:
            if not result.alternatives:
                return
            transcript = result.alternatives[0].transcript
            if result.is_final:
                overwrite_chars = " " * (num_chars_printed - len(transcript))
                appended_text = transcript + overwrite_chars + "\n"
                self.transcription[-1] = appended_text
                self.temp_transcription = self.transcription
                self.temp_transcription.append("")
                os.system("cls" if os.name == "nt" else "clear")
                for line in self.transcription:
                    print(line, end="")
                print("", end="", flush=True)
                self.transcription.append("")
                num_chars_printed = 0
            else:
                partial_transcript += transcript
        if partial_transcript != "":
            overwrite_chars = " " * (num_chars_printed - len(partial_transcript))
            self.temp_transcription[-1] = partial_transcript + (overwrite_chars + "\r")

            os.system("cls" if os.name == "nt" else "clear")
            for line in self.temp_transcription:
                print(line, end="")
            print("", end="", flush=True)
            num_chars_printed = len(partial_transcript) + 3

    def write_final_transcript(self, transcription):
        # Print the final transcript
        os.system("cls" if os.name == "nt" else "clear")
        print("======== Final Transcription ========\n")
        for line in transcription:
            print(line, end="")

    def compute(self, op_input, op_output, context):
        riva_results = op_input.receive("riva_response")
        self.print_streaming(riva_results)
        if op_input.receive("is_complete"):
            final_transcript = self.temp_transcription[1:]
            self.write_final_transcript(final_transcript)
            final_text = "".join(final_transcript)
            op_output.emit(final_text, "final_text")
