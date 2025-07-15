# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from threading import Event
from time import sleep


class ResponseHandler:
    def __init__(self):
        # Queue that the LLM writes responses to
        self.streaming_queue = Queue()
        # Thread-safe flag used to stop LLM response
        self.muted = Event()
        # Used to indicate that the bot's response is complete
        self.stream_end = object()

    def add_response(self, response):
        """
        Adds a response to the streaming queue.
        """
        self.streaming_queue.put(response)

    def end_response(self):
        """
        Puts the end token in the streaming queue.
        """
        return self.streaming_queue.put(self.stream_end)

    def get_response(self):
        """
        Retrieves a response from the streaming queue.
        """
        is_done = False
        response = self.streaming_queue.get(True, 0.25)
        if response is self.stream_end:
            is_done = True
        return is_done, response

    def is_empty(self):
        """
        Returns True if the streaming queue is empty.
        """
        return self.streaming_queue.empty()

    def is_muted(self):
        """
        Returns True if the LLM is muted.
        """
        return self.muted.is_set()

    def reset_queue(self):
        """
        Resets the streaming queue.
        """
        self.streaming_queue = Queue()

    def mute(self):
        """
        Mutes an agent mid-response
        """
        self.muted.set()
        sleep(0.05)
        self.muted.clear()
