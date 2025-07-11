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

from holoscan.core import Operator, OperatorSpec
from llama_cpp.llama import Llama


class LLMOp(Operator):
    # LLM operator that makes requests to a local-LLM
    def __init__(
        self,
        fragment,
        user_prefix="",
        bot_prefix="",
        bot_rule_prefix="",
        system_prompt="",
        request="",
        end_token="",
        *args,
        **kwargs,
    ):
        self.user_prefix = user_prefix
        self.bot_prefix = bot_prefix
        self.bot_rule_prefix = bot_rule_prefix
        self.request = request
        self.system_prompt = system_prompt
        self.end_token = end_token
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("final_text")

    def start(self):
        self.llm = Llama(
            model_path="/workspace/volumes/models/mistral-7b-openorca.Q8_0.gguf",
            n_gpu_layers=99,
            n_ctx=4096,
            n_batch=512,
            n_threads=6,
            verbose=False,
        )

        # Prompt the LLM to ensure it is loaded onto the GPU before the app starts
        self.llm("Q: What is 42?\nA:", max_tokens=8, stop=["Q:", "\n"], temperature=0)

    def _create_llm_prompt(self, text):
        """
        Creates the prompt to summarize medical transcripts
        """
        # Insert the transcribed text into the request
        user_prompt = self.request.format(text=text)
        med_prompt = (
            f"{self.bot_rule_prefix}\n{self.system_prompt}{self.end_token}\n\n"
            f"{self.user_prefix}\n{user_prompt}{self.end_token}\n\n{self.bot_prefix}\n"
        )
        return med_prompt

    def get_llm_response(self, text):
        """
        Sends a request to the LLM and prints the streamed output
        """
        prompt = self._create_llm_prompt(text)

        resData = self.llm(prompt, stop=["</s>", self.end_token], stream=True)

        print("\n======== LLM Summary ========\n")
        for line in resData:
            if line:
                choices = line.get("choices")[0]
                if choices:
                    next_token = choices.get("text")
                    if next_token:
                        print(next_token, end="")
        print("\n\n")

    def compute(self, op_input, op_output, context):
        text = op_input.receive("final_text")
        self.get_llm_response(text)
