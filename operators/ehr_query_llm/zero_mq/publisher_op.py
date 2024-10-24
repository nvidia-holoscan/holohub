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

import logging

from holoscan.core import Fragment, Operator, OperatorSpec

from operators.ehr_query_llm.message_handling import MessageSender


class ZeroMQPublisherOp(Operator):
    """
    Named inputs:
        message: message to be published to 0MQ

    Named output:
        None
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        topic: str,
        queue_endpoint: str = "tcp://*:5556",
        **kwargs,
    ):
        """An operator that checks the messaging queue and emits the message to the request output.

        Args:
            queue_endpoint (str): message queue endpoint
        Raises:
            ValueError: if queue_policy is out of range.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._topic = topic
        self._queue = MessageSender(queue_endpoint)

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("message")

    def compute(self, op_input, op_output, context):
        """
        Pulls the next message in the queue and emits it.
        """
        message = op_input.receive("message")

        if message:
            self._queue.send_json(self._topic, message)
            self._logger.debug("0ZMQ message sent...")
        else:
            self._logger.warn("Empty input")
