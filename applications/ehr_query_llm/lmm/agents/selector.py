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

from .base_agent import Agent
import json


class SelectorAgent(Agent):
    def __init__(self, settings_path, response_handler):
        super().__init__(settings_path, response_handler)

    def parse_response(self, response):
        """
        Parses the SelectorAgent response and extracts the agent selection.
        """
        try:
            # Check if 'response' is a string and decode it into a dictionary
            if isinstance(response, str):
                response_data = json.loads(response)
            else:
                response_data = response

            selected_agent = response_data.get("selection")
            input_text = response_data.get("corrected input")
            return selected_agent, input_text
        except Exception as e:  # Catching a broader range of exceptions
            self._logger.error(f"Failed to parse response: {e}")
            return None, None

    def process_request(self, text, chat_history):

        prompt = self.generate_prompt(text, chat_history)
        self._logger.debug(f"Selector Agent Prompt:\n{prompt}")

        # Send the prompt to the LLM
        response = self.stream_response(prompt, self.grammar, display_output=False)
        # Parse the response

        selected_agent, input_text = self.parse_response(response)
        if selected_agent:
            self._logger.debug(f"Selected agent: {selected_agent}, Input: {input_text}")
        else:
            self._logger.error("No agent selected")
        return selected_agent, input_text