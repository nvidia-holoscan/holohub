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

from queue import Queue
from threading import Thread
from time import sleep
from typing import Generator, Iterable

import numpy as np
import riva.client
import riva.client.proto.riva_asr_pb2 as rasr
import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
import torch
from holoscan.core import Operator, OperatorSpec
from pynput.keyboard import Key, Listener
from riva.client.auth import Auth
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class RivaASROp(Operator):
    """
    Riva ASR Operator that both opens a streaming microphone instance and transcribes this real-time
    audio using Riva. This operator must create the microphone instance as in order to perform real-time
    transcription, Riva uses streaming gRPC calls that requires an audio generator as an input.
    """

    def __init__(
        self,
        fragment,
        auth: Auth,
        cli_args,
        boosted_lm_words=None,
        boosted_lm_score=0,
        *args,
        **kwargs,
    ):
        self.is_speaking = False
        # Used to determine if the transcript should be sent as complete
        self.done_speaking = False
        # Queue used to write ASR responses to in a background thread
        self.streaming_queue = Queue()
        # Used to mute the mic
        self.is_muted = True
        # Flag that gives Riva time to output final prediction
        self.asr_service = riva.client.ASRService(auth)
        self.auth = auth
        self.stub = rasr_srv.RivaSpeechRecognitionStub(auth.channel)
        self.input_queue = AudioQueue()

        streaming_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=cli_args.language_code,
                max_alternatives=1,
                profanity_filter=cli_args.profanity_filter,
                enable_automatic_punctuation=True,
                verbatim_transcripts=not cli_args.no_verbatim_transcripts,
                sample_rate_hertz=cli_args.sample_rate_hz,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        self.streaming_config = streaming_config
        riva.client.add_word_boosting_to_config(
            streaming_config, boosted_lm_words, boosted_lm_score
        )
        self.sample_rate_hz = cli_args.sample_rate_hz
        self.whisper_response = ""
        # Used to store input microphone audio bytes
        self.audio_bytes = bytes()
        self.whisper_pipeline = self._get_whisper_pipeline()
        super().__init__(fragment, *args, **kwargs)

    @staticmethod
    def _get_whisper_pipeline():
        """
        Function that returns a pipeline for Whisper ASR
        """

        device = "cuda:0"
        torch_dtype = torch.float16

        model_id = "openai/whisper-medium.en"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )
        return pipe

    def _get_whisper_transcription(self):
        """
        Function that returns a transcription from Whisper ASR
        """
        audio_data = np.frombuffer(self.audio_bytes, dtype=np.int16).flatten().astype(np.float32)
        audio_data /= np.iinfo(np.int16).max
        audio_data = self._resample_to_16k(audio_data, self.sample_rate_hz)
        transcription = self.whisper_pipeline(audio_data)["text"]
        # Clear the
        self.audio_bytes = bytes()
        return transcription

    @staticmethod
    def _resample_to_16k(audio_data, original_sample_rate):
        """
        Function that resamples audio to 16kHz
        """
        # Calculate the greatest common divisor for up and down sampling rates
        gcd = np.gcd(original_sample_rate, 16000)

        # Calculate up and down sampling factors
        up = 16000 // gcd
        down = original_sample_rate // gcd

        # Resample the audio
        resampled_audio_data = resample_poly(audio_data, up, down)

        return resampled_audio_data

    def streaming_request_generator(
        self, audio_chunks: Iterable[bytes]
    ) -> Generator[rasr.StreamingRecognizeRequest, None, None]:
        """
        Creates an generator object returning streaming ASR requests
        """
        yield rasr.StreamingRecognizeRequest(streaming_config=self.streaming_config)
        for chunk in audio_chunks:
            if self.is_muted:
                # Send empty "audio" bytes if muted
                chunk = bytes(len(chunk))
            yield rasr.StreamingRecognizeRequest(audio_content=chunk)

    def run_riva_asr(self, audio_generator):
        """
        Function used to write Riva responses to the streaming queue
        """
        with audio_generator:
            request_generator = self.streaming_request_generator(audio_chunks=audio_generator)
            # Create the Riva ASR generator
            self.riva_reponse_generator = self.stub.StreamingRecognize(
                request_generator, metadata=self.auth.get_auth_metadata()
            )
            while True:
                riva_response = self.riva_reponse_generator.next()
                self.streaming_queue.put(riva_response)

    def on_space_press(self, key):
        """
        Function call that starts streaming once the space bar is pressed
        """
        if key == Key.space:
            self.is_muted = False
            self.is_speaking = True

    def on_space_release(self, key):
        """
        Function call that stops streaming once the space bar is released
        """
        if key == Key.space:
            # Mute the Mic
            self.is_muted = True
            # Get whisper transcription
            self.whisper_response = self._get_whisper_transcription()
            # Flag that the user is done speaking
            self.done_speaking = True
            self.is_speaking = False

    def receive_audio(self, audio_chunk):
        """
        Function that receives audio from the browser
        """
        self.input_queue.put(audio_chunk)
        if not self.is_muted:
            self.audio_bytes += audio_chunk

    def start(self):
        kwargs = {"audio_generator": self.input_queue}
        riva_thread = Thread(target=self.run_riva_asr, kwargs=kwargs)
        riva_thread.start()

        # Creates & starts a listener for key presses
        self.listener = Listener(on_press=self.on_space_press, on_release=self.on_space_release)
        self.listener.start()

    def setup(self, spec: OperatorSpec):
        spec.output("asr_response")

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        if not self.streaming_queue.empty():
            # Gather Riva responses
            riva_response = self.streaming_queue.get(True, 0.1)
        else:
            # Sleep if no response, otherwise the app is unable to handle the
            # rate of compute() calls
            sleep(0.05)
            riva_response = None

        whisper_response = self.whisper_response
        if whisper_response:
            self.done_speaking = True
            self.whisper_response = ""

        asr_response = {
            "is_speaking": self.is_speaking,
            "done_speaking": self.done_speaking,
            "riva_response": riva_response,
            "whisper_response": whisper_response,
        }
        op_output.emit(asr_response, "asr_response")
        # Reset done_speaking flag
        self.done_speaking = False


class AudioQueue:
    """
    Implement same context manager/iterator interfaces as MicrophoneStream (for ASR.process_audio())
    Credit for this code: https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/llamaspeak/asr.py
    """

    def __init__(self, audio_chunk=1600):
        self.queue = Queue()
        self.audio_chunk = audio_chunk

    def put(self, samples):
        self.queue.put(samples)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __next__(self) -> bytes:
        data = []
        size = 0

        while size <= self.audio_chunk * 2:
            data.append(self.queue.get())
            size += len(data[-1])

        return b"".join(data)

    def __iter__(self):
        return self
