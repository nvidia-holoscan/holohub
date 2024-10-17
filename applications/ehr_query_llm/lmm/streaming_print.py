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

from holoscan.core import Operator, OperatorSpec


class StreamingPrintOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.temp_transcription = [""]
        self.transcription = [""]
        self.waiting_for_final = False
        self.last_final_transcript = ""
        self.whisper_response = ""
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("asr_response")
        # The final output
        spec.output("printer_response")

    def print_streaming(self, response):
        """
        Defines the logic to print the Riva responses as a coherent stream
        Return whether the batch of responses contains the final response
        """
        is_final = False
        num_chars_printed = 0  # used in 'no' additional_info
        if not response or not response.results:
            return
        partial_transcript = ""
        for result in response.results:
            if not result.alternatives:
                return
            transcript = result.alternatives[0].transcript
            if result.is_final:
                is_final = True
                overwrite_chars = " " * (num_chars_printed - len(transcript))
                appended_text = transcript + overwrite_chars + "\n"
                self.transcription[-1] = appended_text
                self.temp_transcription = self.transcription
                self.temp_transcription.append("")
                # os.system('cls' if os.name=='nt' else 'clear')
                # for line in self.transcription:
                #     print(line, end='')
                # print('', end='', flush=True)
                self.transcription.append("")
                num_chars_printed = 0
            else:
                partial_transcript += transcript
        if partial_transcript != "":
            overwrite_chars = " " * (num_chars_printed - len(partial_transcript))
            self.temp_transcription[-1] = partial_transcript + (overwrite_chars + "\r")
            # os.system('cls' if os.name=='nt' else 'clear')
            # for line in self.temp_transcription:
            #     print(line, end='')
            # print('', end='', flush=True)
            num_chars_printed = len(partial_transcript) + 3
        return is_final

    def get_transcript_str(self):
        # Remove empty string from the transcript
        transcript = list(filter(None, self.temp_transcription))
        # Create a string that mirrors what the user sees on the UI
        return "\n".join(transcript)

    def is_prompt_complete(self, transcript, is_final, done_speaking):
        """
        Determines if the current transcript should be sent as a prompt to the model
        Args:
            transcript::str
                The current transcript as a string
            is_final::bool
                Whether the current transcript is flagged as "final" by Riva
            done_speaking::bool
                Whether the user is done speaking
        """
        prompt_complete = False
        # The prompt is complete only when the user is done speaking
        # and Riva sends the "is_final" flag
        if is_final and self.waiting_for_final:
            prompt_complete = True
        # If the user is done speaking and the last transcript flagged
        # as 'final' is the same as the current transcript. This means
        # the user held down the ASR trigger well past when they stopped
        # speaking
        elif done_speaking and self.last_final_transcript == transcript:
            prompt_complete = True

        return prompt_complete

    def compute(self, op_input, op_output, context):
        asr_response = op_input.receive("asr_response")
        done_speaking = asr_response["done_speaking"]
        riva_results = asr_response["riva_response"]
        whisper_response = asr_response["whisper_response"]

        # If speaking is complete, wait for final response
        if done_speaking:
            self.waiting_for_final = True
            self.whisper_response = whisper_response

        # Update the transcript and return whether the response is final
        is_final = self.print_streaming(riva_results)

        # Get the string version of the transcript
        transcript = self.get_transcript_str()
        # strip the transcript of whitespace
        transcript = transcript.strip()

        # If Riva flags the transcript as final save it as last "final" instance
        if is_final:
            self.last_final_transcript = transcript

        prompt_complete = self.is_prompt_complete(transcript, is_final, done_speaking)
        # If the prompt is complete reset the transcript and final flag
        if prompt_complete:
            self.temp_transcription = [""]
            self.transcription = [""]
            self.waiting_for_final = False
            self.last_final_transcript = ""
            transcript = self.whisper_response
            self.whisper_response = ""

        # Get the number of characters in the transcript
        prompt_length = len(transcript.replace(" ", "").replace("\n", ""))
        # If the prompt is complete but there are less than 3 chars
        # reset the flag. 3 is an arbitrary length that likely indicates
        # the user accidentally pressed the space bar
        if prompt_complete and prompt_length < 10:
            prompt_complete = False

        asr_response = {
            "is_speaking": asr_response["is_speaking"],
            "prompt_complete": prompt_complete,
            "asr_transcript": transcript,
        }
        op_output.emit(asr_response, "printer_response")
