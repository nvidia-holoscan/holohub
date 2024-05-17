# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import os
import sys
from typing import Dict

from halo import Halo
from holoscan.core import Application, Operator, OperatorSpec
from openai import APIConnectionError, AuthenticationError, OpenAI

logger = logging.getLogger("httpx")
logger.setLevel(logging.WARN)
logger = logging.getLogger("openai")
logger.setLevel(logging.WARN)
logger = logging.getLogger("NVIDIA_NIM_CHAT")
logging.basicConfig(level=logging.INFO)


class MessageBody:
    def __init__(self, model_params: Dict, user_input: str):
        self.model_params = model_params
        self.user_input = user_input

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_text):
        return MessageBody(**json.loads(json_text))


class OpenAIOperator(Operator):
    def __init__(
        self,
        fragment,
        *args,
        name,
        spinner,
        base_url=None,
        api_key=None,
        **kwargs,
    ):
        self.spinner = spinner
        self.base_url = base_url
        self.api_key = api_key
        self.model_params = dict(kwargs)

        self._reset_chat_history()

        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", None)
        if not self.api_key:
            logger.warning(f"Setting up connection to {base_url} without an API key.")
            logger.warning(
                "Set 'api-key' in the nvidia_nim.yaml config file or set the environment variable 'OPENAI_API_KEY'."
            )
            print("")
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)

        # Need to call the base class constructor last
        super().__init__(fragment, name=name)

    def _reset_chat_history(self):
        self._chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        message = MessageBody.from_json(value)
        if message.user_input == "/c":
            self._reset_chat_history()
            self.spinner.succeed("Chat history cleared.")
        elif message.user_input == "/bye":
            exit()
        else:
            self._chat_history.append({"role": "user", "content": message.user_input})
            self.spinner.stop()

            try:
                completion = self.client.chat.completions.create(
                    messages=self._chat_history, **message.model_params
                )

                chunks = []
                for chunk in completion:
                    if len(chunk.choices) == 0:
                        continue

                    content = chunk.choices[0].delta.content
                    if content is not None:
                        print(content, end="")
                        chunks.append(content)

                self._chat_history.append({"role": "assistant", "content": " ".join(chunks)})
            except (AuthenticationError, APIConnectionError) as e:
                if not self.api_key or "401" in e.args[0]:
                    logger.error(
                        "%s: Hey there! It looks like you might have forgotten to set your API key or the key is invalid. No worries, it happens to the best of us! 😊",
                        e,
                    )
                else:
                    logger.error(str(e))
            except Exception as e:
                print(type(e))
                logger.error("Oops! Wanna try another model? %s", repr(e))
            print(" ")

    def _spinning_cursor(self):
        while True:
            for cursor in "|/-\\":
                yield cursor


class UseInputOp(Operator):
    def __init__(self, fragment, *args, models, selected_model, spinner, **kwargs):
        self.models = models
        self.model_names = [k for k in models.keys()]
        self.spinner = spinner
        self._use_model(selected_model)

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        print(f"\n{self.selected_model['model']}: [/? for help] > ", end="")
        user_input = sys.stdin.readline().strip()

        if user_input == "/?":
            self._print_help()
        elif user_input == "/m":
            self._select_model()
        elif len(user_input) != 0:
            self.spinner.start()
            message = MessageBody(self.selected_model, user_input)
            op_output.emit(message.to_json(), "out")

    def _use_model(self, model):
        self.selected_model = self.models[model]

    def _select_model(self):
        user_input = ""

        while not user_input.isnumeric():
            self._print_models()
            print(f"\n\nSelect a model: 1..{len(self.models)}: ", end="")
            user_input = sys.stdin.readline().strip()
            if user_input.isnumeric():
                user_input = int(user_input)
                if 0 < user_input <= len(self.models):
                    self._use_model(self.model_names[user_input - 1])
                    break
            logger.error(f"'{user_input}' is not a valid option.")
            user_input = ""

    def _print_models(self):
        print("Available Models")
        print("================")
        print(f"{'Option' : ^10} {'Name' : <40}{'Model' : >50}")
        for index, model in enumerate(self.model_names):
            print(f"{index+1 : ^10} {model : <40}{self.models[model]['model'] : >50}")
        print("")

    def _print_help(self):
        print("Chat with NVIDIA NIM!")
        print("==========MENU==========")
        print("/?   Help")
        print("/c   New conversation")
        print("/m   Switch model")
        print("/bye Exit")


class LetsChatWithNIM(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        spinner = Halo(text="thinking...", spinner="dots")

        models = app.kwargs("models")
        selected_model = next(iter(models))

        input_op = UseInputOp(
            self, name="input", models=models, selected_model=selected_model, spinner=spinner
        )
        chat_op = OpenAIOperator(
            self,
            name="chat",
            spinner=spinner,
            **self.kwargs("nim"),
        )

        self.add_flow(input_op, chat_op)


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "nvidia_nim.yaml")
    app = LetsChatWithNIM()
    app.config(config_file)

    try:
        app.run()
    except Exception as e:
        logger.error("Error:", str(e))
