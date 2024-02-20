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

from typing import Generator, Iterable

import riva.client.audio_io
import riva.client.proto.riva_asr_pb2 as rasr
import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
from holoscan.core import Operator, OperatorSpec
from pynput import keyboard
from riva.client.auth import Auth


class RivaASROp(Operator):
    """
    Riva ASR Operator that both opens a streaming microphone instance and transcribes this real-time
    audio using Riva. This operator must create the microphone instance as in order to perform
    real-time transcription, Riva uses streaming gRPC calls that requires an audio generator
    as an input.
    """

    def __init__(self, fragment, auth: Auth, cli_args, *args, **kwargs):
        riva.client.ASRService(auth)
        self.auth = auth
        self.stub = rasr_srv.RivaSpeechRecognitionStub(auth.channel)
        self.first_request = True
        self.is_prompt = False
        self.is_streaming = True
        mic_args = {
            "sample_rate_hz": cli_args.sample_rate_hz,
            "file_streaming_chunk": cli_args.file_streaming_chunk,
            "input_device": cli_args.input_device,
        }
        self.mic_args = mic_args
        streaming_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=cli_args.language_code,
                max_alternatives=1,
                profanity_filter=cli_args.profanity_filter,
                enable_automatic_punctuation=cli_args.automatic_punctuation,
                verbatim_transcripts=not cli_args.no_verbatim_transcripts,
                sample_rate_hertz=cli_args.sample_rate_hz,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        self.streaming_config = streaming_config
        riva.client.add_word_boosting_to_config(
            streaming_config, cli_args.boosted_lm_words, cli_args.boosted_lm_score
        )
        super().__init__(fragment, *args, **kwargs)

    def streaming_request_generator(
        self, audio_chunks: Iterable[bytes]
    ) -> Generator[rasr.StreamingRecognizeRequest, None, None]:
        """
        Creates an generator object returning streaming ASR requests
        """
        yield rasr.StreamingRecognizeRequest(streaming_config=self.streaming_config)
        for chunk in audio_chunks:
            yield rasr.StreamingRecognizeRequest(audio_content=chunk)

    def on_key_press(self, key):
        """
        Function call that stops streaming once the 'x' key is pressed
        """
        if key == keyboard.KeyCode.from_char("x"):
            self.is_streaming = False

    def start(self):
        # Create the Riva microphone stream
        self.mic_instance = riva.client.audio_io.MicrophoneStream(
            self.mic_args["sample_rate_hz"],
            self.mic_args["file_streaming_chunk"],
            device=self.mic_args["input_device"],
        )
        # Start the micropohone stream
        self.mic_stream = self.mic_instance.__enter__()
        request_generator = self.streaming_request_generator(audio_chunks=self.mic_stream)
        # Create the Riva ASR generator
        self.riva_reponse_generator = self.stub.StreamingRecognize(
            request_generator, metadata=self.auth.get_auth_metadata()
        )

        # Creates & starts a listener for key presses
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def setup(self, spec: OperatorSpec):
        spec.output("riva_response")
        spec.output("is_complete")

    def stop(self):
        # Stop the mic streaming & the key listener
        self.mic_instance.close()
        self.listener.stop()
        pass

    def compute(self, op_input, op_output, context):
        if self.is_streaming:
            riva_response = self.riva_reponse_generator.next()
        else:
            riva_response = None
        op_output.emit(riva_response, "riva_response")
        # Let the downstream Op know when transcription is complete
        is_complete = not self.is_streaming
        op_output.emit(is_complete, "is_complete")
        if is_complete:
            # Will allow downstream operators to run but will not begin
            # another execution cycle after
            self.conditions.get("stop_execution_condition").disable_tick()
