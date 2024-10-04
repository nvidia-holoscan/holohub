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
import os
import yaml
import re
from pathlib import Path


class ChatHistory:
    def __init__(self):
        # Keeps track of the chat history. The format is a list of string tuples
        # where index 0 is the user, and index 1 is the bot
        self.history = [[""]]
        # Keep track of whose turn it is in the conversation
        self.current_responder = "user"
        # Index used to get the current user message from history
        self.user_index = -1
        # Index used to get the current bot message from history
        self.bot_index = -1

    def add_image(self, image_html):
        # Add a place for the bot to respond with the image
        if len(self.history[self.bot_index]) < 2:
                self.history[self.bot_index].append("")
        # Add a pair of responses to the history if the last
        # set of responses is complete
        elif self.history[self.bot_index][1] != "":
            self.history.append(["", ""])
        self.history[-1][1] = image_html
        self.history.append(["", ""])

    def update_chat_history(self, llm_complete, llm_response, prompt_complete, asr_transcript):
        if self.current_responder == "user" and llm_response:
            # Create an empty string for the bot to write to
            # if there isn't one already
            if len(self.history[self.bot_index]) < 2:
                self.history[self.bot_index].append("")
            # Add a pair of responses to the history if the last
            # set of responses is complete
            elif self.history[self.bot_index][1] != "":
                self.history.append(["", ""])
            # Change the responder to the bot
            self.current_responder = "bot"

        if prompt_complete and self.current_responder == "bot":
            # If its the bots turn and theres a llm response add the llm
            # response
            if llm_response:
                self.history[self.bot_index][1] = llm_response
            # Add a location for the bot to respond to in the same
            # index as the current user index
            if len(self.history[-1]) < 2:
                self.history[-1].append("")
            # If the bot was responding to a previous prompt update
            # the bot index to the most recent
            if self.bot_index == -2:
                self.bot_index = -1

        elif prompt_complete == True:
            # If its the user's turn and they finished speaking update
            # the current responder
            self.current_responder = "bot"
            # Add a string for the llm to write to
            if len(self.history[-1]) < 2:
                self.history[-1].append("")
        # If the bot is responding, but the user has started talking, then allow the
        # bot to continue responding to the past prompt
        elif self.bot_index == -1 and self.current_responder == "bot" and asr_transcript:
            self.history.append([""])
            self.bot_index = -2

        # If the llm is responding and the user hasn't completed their prompt
        if llm_response and self.current_responder == "bot" and not prompt_complete:
            # Update the bot's response
            self.history[self.bot_index][1] = llm_response
        if llm_complete:
            # If the bot is done responding reset the bot index or
            # add a string for the user to then write to
            if self.bot_index == -2:
                    self.bot_index = -1
            else:
                self.history.append([""])
            self.current_responder = "user"

        # If there is a new ASR transcript update the user's chat bubble
        if asr_transcript:
            self.history[self.user_index][0] = asr_transcript
        
        # Only keep last 15 turns to reduce memory usage
        if len(self.history) > 15:
            self.history = self.history[-15:]

    def to_list(self):
        # Returns the chat history as a list
        return self.history

def get_tool_definitions(directory="tools"):
    """
    Builds tool definitions using the tool .yaml files
    """
    final_string = ""
    directory_path = Path(os.path.join(Path(__file__).parent, directory))
    for file in directory_path.iterdir():
        if str(file).endswith(".yaml"):
            with open(str(file), 'r') as f:
                data = yaml.safe_load(f)
                name = data.get('name', 'N/A')
                description = data.get('description', 'N/A')
                usage = data.get('usage', 'N/A')
                final_string += f"name: {name}\ndescription: {description}\n{usage}\n"

    return final_string
