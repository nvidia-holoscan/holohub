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
import subprocess
import sys

import openai
import whisper
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level

sys.stderr = sys.stdout


class WhisperOp(Operator):
    def __init__(self, *args, model_name="base", file_name="example.wav", **kwargs):
        self.count = 0
        self.model_name = model_name
        self.file_name = file_name
        self.model = whisper.load_model(self.model_name)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("text")

    def compute(self, op_input, op_output, context):
        print("Starting Audio Transcription...")
        if self.file_name.endswith(".mp4"):
            audio_file = self.extract_audio_from_video(self.file_name)
        else:
            audio_file = self.file_name

        output = self.model.transcribe(audio_file)
        text = output["text"]
        self.count += 1

        print("Audio Transcription Finished...")
        op_output.emit(text, "text")

    def extract_audio_from_video(self, video_file):
        audio_file = video_file.replace(".mp4", ".wav")
        subprocess.call(["ffmpeg", "-i", video_file, audio_file])
        return audio_file


class ChatGPTOp(Operator):
    def __init__(self, *args, model, context, api_key=None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.context = context
        openai.api_key = api_key
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("text")
        spec.output("gpt_answers")

    def get_gpt_response(self, text):
        prompt = f"Transcript from Radiologist: {text}\n Request(s): {self.context}"
        print("Making LLM API Call...")
        answer = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a veteran radiologist, who can answer any medical "
                    "related question.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return answer["choices"][0]["message"]["content"]

    def compute(self, op_input, op_output, context):
        text = op_input.receive("text")
        answers = self.get_gpt_response(text)

        op_output.emit(answers, "gpt_answers")


class PrintTextOp(Operator):
    """Print the received text to the terminal."""

    def __init__(self, *args, prompt="TRANSCRIPTION: ", **kwargs):
        self.prompt = prompt
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("text")

    def compute(self, op_input, op_output, context):
        text = op_input.receive("text")
        print(self.prompt, text + "\n")


class STT_To_LLM(Application):
    """Speech-to-text transcription and Large Language Model application.

    Transcribes an audio file to text, uses an LLM to perform requests specified in yaml file.
    """

    def compose(self):
        stt = WhisperOp(
            self, CountCondition(self, count=1), name="whisper", **self.kwargs("WhisperOp")
        )
        llm = ChatGPTOp(self, name="gpt3", **self.kwargs("ChatGPTOp"))
        print_stt = PrintTextOp(self, name="print_whisper", prompt="\nAudio Transcription: \n")
        print_llm_answers = PrintTextOp(self, name="gpt_answers", prompt="\nLLM Response: \n")

        self.add_flow(stt, llm)
        self.add_flow(stt, print_stt)
        self.add_flow(llm, print_llm_answers, {("gpt_answers", "text")})


if __name__ == "__main__":
    set_log_level(LogLevel.WARN)

    app = STT_To_LLM()
    config_file = os.path.join(os.path.dirname(__file__), "stt_to_nlp.yaml")
    app.config(config_file)
    app.run()
