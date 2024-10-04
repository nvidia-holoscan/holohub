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
from applications.holoscrub.lmm.rag.ehr.create_ehr_db_local import create_ehr_database
import json

class EHRBuilderAgent(Agent):
        def __init__(self, settings_path, response_handler):
            super().__init__(settings_path, response_handler)

        def generate_prompt(self, text):
            """
            Generate a prompt for the LLM based on the given text, chat history, and agent configuration,
            including information about the available tools.
            """
            # Create the system prompt component
            system_prompt = f"{self.bot_rule_prefix}\n{self.agent_prompt}\n{self.end_token}"
            # Create the system prompt component
            user_prompt = f"\n{self.user_prefix}\n{text}\n{self.end_token}"
            # Calulate the token usage of the system and user prompts
            prompt = system_prompt + user_prompt
            prompt += f"\n{self.bot_prefix}\n"
            return prompt

        def process_request(self, text):
            """
            Processes a request related to EHR building tasks using the LLM.
            """
            # Generate prompt
            prompt = self.generate_prompt(text)
            # Send the prompt to the LLM and get the response
            self._logger.debug(f"EHR Builder Agent Prompt:\n{prompt}")

            response = self.stream_response(prompt, self.grammar)
            
            json_response = json.loads(response.replace("\n", "\\n"))
            can_build = json_response.get("can_build", False)
            if can_build:
                time_to_build = create_ehr_database()
                response = '{"name": "EHRBuilderAgent", "response": "Completed building the EHR database. The database was built in ' + str(time_to_build) + ' seconds."}'

                self.response_handler.add_response(response)
                self.response_handler.end_response()

            # Return the complete response
            return response