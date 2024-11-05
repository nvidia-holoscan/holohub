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

from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

from operators.ehr_query_llm.message_handling import MessageReceiver


class ZeroMQSubscriberOp(Operator):
    """
    Named inputs:
        none

    Named output:
        request: a request message received from the 0MQ message queue
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        topic: str,
        queue_endpoint: str = "tcp://localhost:5556",
        blocking: bool = False,
        **kwargs,
    ):
        """An operator that checks the messaging queue and emits the message to the request output.

        Args:
            topic (str): name of the topic to filter. Single for now, to be replaced
            queue_endpoint (str): message queue endpoint
            blocking (bool): if a blocking receive. Defaults to False due to existing code assumption

        Raises:
            ValueError: if queue_policy is out of range (is only one of the exceptions)
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        self._queue = MessageReceiver(topic, queue_endpoint)
        self._blocking = blocking

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("request").condition(ConditionType.NONE)  # Downstream receiver optional.

    def compute(self, op_input, op_output, context):
        """
        Pulls the next message in the queue and emits it.
        """
        request_str = self._queue.receive_json(self._blocking)
        if not request_str:
            return

        op_output.emit(request_str, "request")
        self._logger.debug("Received request message sent downstream...")
