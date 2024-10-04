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

import json
import re
import time
from queue import Queue
from threading import Event, Thread
from time import sleep

import riva.client
from holoscan.core import Operator, OperatorSpec
from riva.client.auth import Auth


class RivaTTSOp(Operator):
    """
    Riva TTS Operator that uses the python client library to make TTS
    requests to the the Riva server.
    """

    def __init__(self, fragment, auth: Auth, cli_args, *args, **kwargs):
        self.cli_args = cli_args
        self.service = riva.client.SpeechSynthesisService(auth)
        # Keeps track of streamed responses
        self._last_response = ""
        # Queue for streaming TTS
        self.streaming_queue = Queue()
        # Event for muting TTS
        self.is_muted = Event()
        # Event for checking if TTS is currently responding
        self.is_responding = False
        # List to keep track of previous sentences that were spoken (Only applies to agents with a "response" field)
        # This is needed to stream TTS before the LLM is done responding
        self._previous_sentences = []
        # Keep track of the last TTS response so we don't repeat it
        self._last_tts = None
        super().__init__(fragment, *args, **kwargs)

    def set_audio_callback(self, callback):
        self.on_tts_audio = callback

    def start(self):
        tts_thread = Thread(target=self.create_tts)
        tts_thread.start()

    def set_voice(self, voice):
        self.cli_args.voice = voice

    def setup(self, spec: OperatorSpec):
        spec.input("agent_response")

    def stop(self):
        pass

    def create_tts(self):
        while True:
            agent_response = self.streaming_queue.get()
            # Split response into multi-line chunks so TTS isn't overwhelmed
            for agent_chunk in agent_response.split("\n"):
                # Skip empty chunks
                if not agent_chunk.strip():
                    continue
                # Note: Default encoding is LINEAR_PCM (Uncompressed 16-bit signed little-endian samples)
                # Note: only 1 audio channel is used
                bit_depth = 16
                responses = self.service.synthesize_online(
                    agent_chunk,
                    self.cli_args.voice,
                    self.cli_args.language_code,
                    sample_rate_hz=self.cli_args.sample_rate_hz,
                )
                self.is_responding = True
                # Size per second = sample_rate_hz * bit_depth * channels / 8
                # Make the chunk sizes 1/4 second
                chunk_size = self.cli_args.sample_rate_hz * bit_depth // (4 * 8)
                for resp in responses:
                    for chunk in range(0, (len(resp.audio) + chunk_size) // chunk_size, 1):
                        audio = resp.audio[chunk * chunk_size : (chunk + 1) * chunk_size]
                        time_start = time.perf_counter()
                        self.on_tts_audio(payload=audio, type=2)
                        # Break from chunk loop if TTS is being muted
                        if self.is_muted.is_set():
                            break
                        # Sleep buffer to keep TTS smooth
                        sleep_buffer = 0.01
                        # Calculate time it will take to play the audio
                        time_sleep = (
                            (len(audio) / (self.cli_args.sample_rate_hz * bit_depth / 8))
                            - (time.perf_counter() - time_start)
                        ) - sleep_buffer
                        if time_sleep > 0.001:
                            sleep(time_sleep)
                    # Break from streaming response loop if TTS is being muted
                    if self.is_muted.is_set():
                        break
                # If muted, reset previous sentences, clear the queue, and begin waiting for the next request
                if self.is_muted.is_set():
                    self._previous_sentences = []
                    while not self.streaming_queue.empty():
                        self.streaming_queue.get()
                    break
            # Reset the mute event and the responding flag
            self.is_muted.clear()
            self.is_responding = False

    def compute(self, op_input, op_output, context):
        llm_emit = op_input.receive("agent_response")
        is_done = llm_emit["is_done"]
        agent_response = llm_emit["agent_response"]
        is_speaking = llm_emit["is_speaking"]

        # If the user is interrupting mute the TTS
        if self.is_responding and is_speaking:
            self.is_muted.set()

        # Will receive a valid TTS string or None
        agent_response = self.make_tts_friendly(agent_response)

        # Check if there is a response or if the LLM is done
        if agent_response or is_done:
            # If the LLM is done but no response is provided use the last valid response
            if not agent_response:
                agent_response = self._last_response
            # Add the response to the queue if it's not the same as the last response
            if agent_response != self._last_tts:
                self.streaming_queue.put(agent_response)
                self._last_tts = agent_response
            # Reset the previous sentences if the LLM is done
            if is_done:
                self._previous_sentences = []
                self._last_tts = None

        # Keep track of the last response
        if agent_response:
            self._last_response = agent_response

    def make_tts_friendly(self, agent_response):
        """
        Converts the JSON-like streamed response to a TTS-friendly string.
        """
        sentences = None
        try:
            # Parse agents with a "response" field before the LLM is complete
            # (Results in lower latency for TTS streaming)
            response_agents = ["ChatAgent", "EHRAgent", "EHRBuilderAgent"]
            for agent in response_agents:
                if agent in agent_response:
                    sentences = self.parse_streamed_response(agent_response)
                    # Remove any sentences that have already been spoken
                    sentences = [
                        sentence
                        for sentence in sentences
                        if sentence not in self._previous_sentences
                    ]
                    # Add the new sentences to the previous sentences
                    self._previous_sentences += sentences
                    # Crease single string from the sentences
                    tts_response = "\n".join(sentences)
                    tts_response = self.replace_abbreviations(tts_response)
                    return tts_response

            # Parse agents without a "response" field when the LLM is complete
            # agent_response = agent_response.replace("\n", "\\n")
            # json_response = json.loads(agent_response)

        except json.JSONDecodeError:
            # If it's not valid JSON, return None
            return None
        except Exception as e:
            # Log other exceptions for debugging
            print(f"Error in make_tts_friendly: {e}")
            return None

    @staticmethod
    def replace_abbreviations(text):
        """
        Replaces common abbreviations with their full forms to help Riva TTS
        """
        text = re.sub(r"(\d+)mm", r"\1 millimeter", text)
        text = re.sub(r"(\d+)cm", r"\1 centimeter", text)
        text = re.sub(r"(\d+)mg/dL", r"\1 milligrams per deciliter", text)
        return text

    @staticmethod
    def parse_streamed_response(streamed_text, min_sentence_length=5):
        """
        Parses streamed text to extract complete sentences from the 'response' JSON field,
        and checks if the end of the "response" field is encountered by looking for
        patterns indicating the actual end of the field.

        Parameters:
        - streamed_text (str): The streamed text in JSON-like format.
        - min_sentence_length (int): Minimum length for a sentence to be considered valid.

        Returns:
        - sentences (list): A list of complete sentences extracted from the 'response' part.
        """
        # Extracting text after '"response" : "'
        match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)', streamed_text, re.DOTALL)
        if not match:
            return []  # Return an empty list if no match is found
        response_text = match.group(1)

        # Check for patterns indicating the end of the "response" field
        end_of_response_encountered = streamed_text.strip().endswith("}")
        # Removing everything after the last legitimate ending if end_of_response_encountered
        if end_of_response_encountered:
            response_text = re.sub(r'"\s*(,|})?$', "", response_text)
        # Splitting on periods to get sentences
        potential_sentences = response_text.split(".")
        # Removing the last element if it's likely an incomplete sentence and not at the end of the response
        valid_last_sentence = potential_sentences[-1].strip().endswith(".")
        # Filtering sentences based on length
        sentences = []
        for sentence in potential_sentences:
            sentence = sentence.strip()
            if len(sentence) >= min_sentence_length or not sentences:
                sentences.append(sentence)
            else:
                sentences[-1] += "\n" + sentence

        # If the last sentence is not complete and the end of the response was not encountered, remove it
        if sentences and not valid_last_sentence and not end_of_response_encountered:
            sentences = sentences[:-1]

        return sentences
