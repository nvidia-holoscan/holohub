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
from threading import Event, Thread
from queue import Queue
import json
import logging
import os
from time import time
import datetime

from time import sleep
from utils.response_handler import ResponseHandler
from utils.chat_utils import ChatHistory

from holoscan.core import Operator, OperatorSpec

from operators.holoscrub.message_handling import MessageSender, MessageReceiver
from agents.selector import SelectorAgent
from agents.chat import ChatAgent

from agents.ehr import EHRAgent

from agents.ehr_builder import EHRBuilderAgent

from agents.fhirchat import FHIRChatAgent


class AgentFrameworkOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        **kwargs,
    ):
        self.last_response = None
        self.is_responding = False
        self.streaming_queue = Queue()
        self.rag_documents = None
        self.sender = MessageSender("tcp://*:5555")
        self.image_receiver = MessageReceiver(topic="primary_app_image", endpoint="tcp://localhost:5560")
        self.frame_request_topic = "primary_app_request"
        self.frame_request_timeout = 5
        # Determines which agents are available
        self.episode_num = 1

        # instantiate agents
        self.AGENTS_BASE_PATH = "/workspace/holohub/applications/holoscrub/lmm/agents_configs"

        # instantiate response handler
        self.response_handler = ResponseHandler()

        # selector agent
        selector_path = self.get_agent_settings_path("selector")
        self.selector_agent = SelectorAgent(selector_path, self.response_handler)

        # chat agent
        chat_agent_path = self.get_agent_settings_path("chat")
        self.chat_agent = ChatAgent(chat_agent_path, self.response_handler)

        # EHRBuilder agent
        ehr_builder_agent_path = self.get_agent_settings_path("ehr_builder")
        self.ehr_builder_agent = EHRBuilderAgent(ehr_builder_agent_path, self.response_handler) 

        # EHR agent
        ehr_agent_path = self.get_agent_settings_path("ehr")
        self.ehr_agent = EHRAgent(ehr_agent_path, self.response_handler)
        
        # create chat history
        self.chat_history = ChatHistory()
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(fragment, *args, **kwargs)

    def get_agent_settings_path(self, agent_name):
        """
        Constructs the full path to the settings file for a given agent.
        """
        return os.path.join(self.AGENTS_BASE_PATH, f"{agent_name}.yaml")

    def setup(self, spec: OperatorSpec):
        spec.input("printer_response")
        spec.output("chat_history")
        spec.output("agent_response")

    def process_selector_request(self, asr_text):
        """Function for Selector Agent to choose Downstream Agent and Start Thread"""
        selected_agent_name, corrected_text = self.selector_agent.process_request(asr_text, self.chat_history.to_list())
        self._logger.debug(f"Original Text: {asr_text}")
        self._logger.debug(f"Corrected Text: {corrected_text}")
        self._logger.debug(f"Selected Agent: {selected_agent_name}")
        if selected_agent_name:
            self.process_agent_request(selected_agent_name, corrected_text)
        else:
            self._logger.error("Invalid agent selection")
        self.is_responding = False

    def process_agent_request(self, selected_agent_name, asr_text):
        """Function to process request in a separate thread."""
        try:
            if selected_agent_name == "EHRBuilderAgent":
                response = self.ehr_builder_agent.process_request(asr_text)
            elif selected_agent_name == "EHRAgent":
                response, documents = self.ehr_agent.process_request(asr_text, self.chat_history.to_list())
                self.rag_documents = documents
            else:
                response = "Invalid agent selection"
                self._logger.error(response)
        except Exception as e:
            self._logger.error(f"Error processing agent request: {e}")


    def compute(self, op_input, op_output, context):
        printer_response = op_input.receive("printer_response")
        prompt_complete = printer_response["prompt_complete"]
        asr_text = printer_response["asr_transcript"]
        is_speaking = printer_response.get("is_speaking", False)

        is_done = False
        # The bot is responding if the stream isn't empty or the flag is still set from the
        # llm response loop
        self.is_responding = not self.response_handler.is_empty() or self.is_responding

        agent_response = ""
        # Process input
        if self.is_responding:
            # Loop to get the most recent response
            while not self.response_handler.is_empty():
                is_done, tmp_response = self.response_handler.get_response()
                if not is_done:
                    agent_response = tmp_response
                    if agent_response:
                        self.last_response = agent_response
                else:
                    self.sender.send_json("rendering", {"state": "listening"})

        # Update chat history
        self.chat_history.update_chat_history(is_done, agent_response, prompt_complete, asr_text)

        # Mute the Agent if necessary
        if self.is_responding and is_speaking:
            self.response_handler.mute()

        if prompt_complete:
            self.sender.send_json("rendering", {"state": "processing"})
            # Prompt is complete. Start agent framework thread with selector agent
            agent_thread = Thread(target=self.process_selector_request, args=(asr_text,))
            agent_thread.start()

        op_output.emit(self.chat_history.to_list(), "chat_history")

        # Emit the updated response
        agent_response = {
            "is_done": is_done,
            "agent_response": agent_response,
            "is_speaking": is_speaking,
            "chat_history": self.chat_history.to_list(),  # Optionally emit the chat history
            "rag_documents": self.rag_documents
        }
        op_output.emit(agent_response, "agent_response")

    def publish(self, agent_response):
        """
        Publishes any tool Chosen by an Agent
        """
        try: 
            json_response = json.loads(agent_response)
            if json_response.get("name") == "chat":
                return
            else:
                topic = json_response.get("name")
                output_msg = {}
                for key, value in json_response.items():
                    if key != "name":
                        output_msg[key] = value
                self.sender.send_json(topic, output_msg)
                self._logger.debug(f"Topic published: {topic}\nOutput Message: {output_msg}")
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON decoding error in publish: {e}")
        except Exception as e:
            self._logger.error(f"Error in publish: {e}")


    def reset_history(self):
        self.chat_history = ChatHistory()
        self._logger.info(f"Reset chat history")


    def _get_video_frame(self):
        self.sender.send_json(self.frame_request_topic, "Image please")
        self._logger.debug("0MQ frame request sent...")
        frame_response = self.image_receiver.receive_json()
        image_b64 = frame_response.get("image_b64", None)
        tool_labels = frame_response.get("tool_labels", {})
        start_time = datetime.datetime.now()
        while image_b64 == None:
            if (datetime.datetime.now() - start_time).total_seconds() > self.frame_request_timeout:
                break
            sleep(0.1)
            json_message = self.image_receiver.receive_json()
            image_b64 = json_message.get("image_b64", None)
            tool_labels = json_message.get("tool_labels", {})
        if image_b64:
            self._logger.debug("0MQ frame received")
            self._logger.debug(f"Instruments received: {tool_labels}")
            self._logger.debug(f"Image received: {image_b64[:100]}")
        else:
            self._logger.error("0MQ frame not received")
        self._logger.debug("0MQ message received")
        visual_info = {
            "image_b64": image_b64,
            "tool_labels": tool_labels
        }
        return visual_info