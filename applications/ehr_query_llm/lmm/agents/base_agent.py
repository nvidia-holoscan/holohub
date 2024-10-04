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

from abc import ABC, abstractmethod
import requests
from time import sleep
import json
import urllib
import logging
import yaml
import time
import tiktoken
from threading import Lock


class Agent(ABC):
    # Used to lock the LLM for concurrent requests
    _llm_lock = Lock()
    # Used to lock the LMM for concurrent requests
    _lmm_lock = Lock()
    def __init__(self, settings_path, response_handler, agent_key=None):
        self.load_settings(settings_path)
        self.response_handler = response_handler
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._wait_for_server()

    def load_settings(self, settings_path, agent_key=None):
        """
        Load and apply agent settings from a YAML file.
        """
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)

        # If an agent key is provided, use nested settings; otherwise, use top-level settings
        self.agent_settings = settings[agent_key] if agent_key else settings

        # Extracting common configuration for all agents
        self.description = self.agent_settings.get('description', '')
        self.max_prompt_tokens = self.agent_settings.get('max_prompt_tokens', 3000)
        self.ctx_length = self.agent_settings.get('ctx_length', '')
        # Multi-line prompt requires a .strip() to remove the newline character
        self.agent_prompt = self.agent_settings.get('agent_prompt', '').strip()
        self.user_prefix = self.agent_settings.get('user_prefix', '')
        self.bot_prefix = self.agent_settings.get('bot_prefix', '')
        self.bot_rule_prefix = self.agent_settings.get('bot_rule_prefix', '')
        self.end_token = self.agent_settings.get('end_token', '')
        self.grammar = self.agent_settings.get('grammar', None)
        self.publish_settings = self.agent_settings.get('publish', {})
        # Get the url for the LLM api
        self.llm_url = self.agent_settings.get('llm_url')

        # Additional handling for tool-specific settings if needed
        self.tools = self.agent_settings.get('tools', {})

    def stream_response(self, prompt, grammar, temperature=0, display_output=True, allow_muting=True):
        """
        Stream a response from the LLM and optionally write to the output queue.
        This method will attempt to connect to the LLM server for up to 30 seconds.
        """
        lock_acquired = False
        start_time = time.time()
        # Attempt making the request for up to 30 seconds
        # this is needed in instances of concurrent requests (often due to summarization)
        while time.time() - start_time < 30:
            try:
                # Attempt to acquire the lock for the LLM
                # If not, the ConnectionRefusedError will be raised
                if Agent._llm_lock.acquire(blocking=True, timeout=30):
                    lock_acquired = True
                    # Llama.cpp-specific request data
                    request_data = {
                        "prompt": prompt,
                        "grammar": grammar,
                        "temperature": temperature,
                        "max_tokens": self.ctx_length,
                        "stop": ["</s>", self.end_token],
                        "n_keep": -1,
                        "stream": True
                    }
                    resData = requests.request(
                        "POST",
                        urllib.parse.urljoin(self.llm_url, "/completion"),
                        data=json.dumps(request_data),
                        stream=True,
                    )
                    if display_output:
                        self.response_handler.reset_queue()
                    response = ""
                    for line in resData.iter_lines():
                        if allow_muting and self.response_handler.is_muted():
                            return None
                        if line:
                            decoded_line = line.decode("utf-8")
                            next_token = json.loads(decoded_line[6:]).get("content")
                            # Llama.cpp returns a "slot unavailable" message when the slot is unavailable
                            # as opposed to raising an exception
                            if next_token == 'slot unavailable':
                                raise ConnectionRefusedError("Slot unavailable")
                            response += next_token
                            if display_output:
                                self.response_handler.add_response(response)
                    if display_output:
                        self.response_handler.end_response()

                    return response
            except ConnectionRefusedError:
                time.sleep(1)
            finally:
                if lock_acquired:
                    Agent._llm_lock.release()
        raise ConnectionRefusedError("Llama.cpp slot unavailable after 30 seconds")
    
    def stream_image_response(self, prompt, image_b64, grammar, temperature=0, display_output=True, allow_muting=True):
        """
        Annotation agent specific request to the annotation LMM
        """
        lock_acquired = False
        start_time = time.time()
        # Attempt making the request for up to 30 seconds
        # this is needed in instances of concurrent requests (often due to summarization)
        while time.time() - start_time < 30:
            try:
                # Attempt to acquire the lock for the LLM
                # If not, the ConnectionRefusedError will be raised
                if Agent._lmm_lock.acquire(blocking=True, timeout=30):
                    lock_acquired = True
                    # Llama-specific request data with image data
                    image_data = [{
                        "data": image_b64,
                        "id": 0
                    }]

                    request_data = {
                        "prompt": prompt,
                        "image_data": image_data,
                        "grammar": grammar,
                        "temperature": temperature,
                        "max_tokens": self.ctx_length,
                        "stop": ["</s>", self.end_token],
                        "n_keep": -1,
                        "stream": True,
                    }
                    resData = requests.request(
                        "POST",
                        urllib.parse.urljoin(self.llm_url, "/completion"),
                        data=json.dumps(request_data),
                        stream=True,
                    )
                    if display_output:
                        self.response_handler.reset_queue()
                    response = ""
                    for line in resData.iter_lines():
                        if allow_muting and self.response_handler.is_muted():
                            return None
                        if line:
                            decoded_line = line.decode("utf-8")
                            next_token = json.loads(decoded_line[6:]).get("content")
                            response += next_token
                            if display_output:
                                self.response_handler.add_response(response)
                    if display_output:
                        self.response_handler.end_response()

                    return response
            except ConnectionRefusedError:
                time.sleep(1)
            finally:
                if lock_acquired:
                    Agent._lmm_lock.release()
        raise ConnectionRefusedError("Llama.cpp slot unavailable after 30 seconds")

    def generate_prompt(self, text, chat_history):
        """
        Generate a prompt for the LLM/LMM based on the given text, chat history, and agent configuration.
        """
        # Create the system prompt component
        system_prompt = f"{self.bot_rule_prefix}\n{self.agent_prompt}\n{self.end_token}"
        # Create the system prompt component
        user_prompt = f"\n{self.user_prefix}\n{text}\n{self.end_token}"
        # Calulate the token usage of the system and user prompts
        token_usage = self.calculate_token_usage(system_prompt + user_prompt)
        # Create the chat history component without exceeding the maximum prompt tokens
        chat_prompt = self.create_conversation_str(chat_history, token_usage)
        prompt = system_prompt + chat_prompt + user_prompt
        prompt += f"\n{self.bot_prefix}\n"

        return prompt

    def create_conversation_str(self, chat_history, token_usage, conversation_length=2):
        total_tokens = token_usage
        msg_hist = []
        # Remove the last message from the chat history since it is the current user message
        # then reverse the list so that the most recent messages are first
        for user_msg, bot_msg in chat_history[:-1][-conversation_length:][::-1]:
            if bot_msg:
                bot_msg_str = f"\n{self.bot_prefix}\n{bot_msg}\n{self.end_token}"
                bot_tokens = self.calculate_token_usage(bot_msg_str)
                if total_tokens + bot_tokens > self.max_prompt_tokens:
                    break
                else:
                    total_tokens += bot_tokens
                    msg_hist.append(bot_msg_str)
            if user_msg:
                user_msg_str = f"\n{self.user_prefix}\n{user_msg}\n{self.end_token}"
                user_tokens = self.calculate_token_usage(user_msg_str)
                if total_tokens + user_tokens > self.max_prompt_tokens:
                    break
                else:
                    total_tokens += user_tokens
                    msg_hist.append(user_msg_str)

        msg_hist = "".join(msg_hist[::-1])

        return msg_hist

    def calculate_token_usage(self, text):
        """
        Calculates the number of tokens used by the given text
        """
        return len(self.tokenizer.encode(text))

    def _wait_for_server(self, timeout=60):
        """
        Method that attempts to connect to the llama.cpp server
        for up to timeout until throwing an exception
        """
        attempts = 0
        sleep_time = 5
        while attempts < timeout / sleep_time:
            try:
                response = requests.get(self.llm_url)
                # Check for a successful response status code (e.g., 200 OK)
                if response.status_code == 200:
                    self._logger.debug(f"{type(self).__name__} connected to Llama.cpp server at {self.llm_url}")
                    return
            except requests.ConnectionError:
                sleep(sleep_time)
                print("Attempting to connect to LLM server...")
            attempts += 1
        raise requests.ConnectionError(f"Unable to connected to Llama.cpp server at {self.llm_url}")

    def _get_tool_str(self, tool_labels):
            tool_str = ""
            if tool_labels:
                for name, coords in tool_labels.items():
                    # only use 3 decimal places
                    tool_str += f"{name}: [{coords[0]:.3f}, {coords[1]:.3f}], "
            if tool_str:
                tool_str = tool_str[:-2]
            return tool_str

    @abstractmethod
    def process_request(self, input_data, chat_history):
        """
        Process a request. To be implemented per agent.
        """
        pass

    def append_json_to_file(self, json_object, file_path):
        """
        Appends JSON annotations to the meta file
        """
        try:
            # Read the existing content of the file
            with open(file_path, 'r') as file:
                content = file.read().rstrip('\n\n,] ') + ','

            # If the file is not empty and already contains a JSON array
            if content:
                with open(file_path, 'w') as file:
                    file.write(content + '\n\t' + json.dumps(json_object) + '\n]\n')
            else:
                # If the file is empty, start a new JSON array
                with open(file_path, 'w') as file:
                    file.write('[' + '\n\t' + json.dumps(json_object) + '\n]\n')
        except FileNotFoundError:
            # If the file does not exist, create it and add the JSON object
            with open(file_path, 'w') as file:
                file.write('[' + '\n\t' + json.dumps(json_object) + '\n]\n')
