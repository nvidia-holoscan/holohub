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

import threading

from holoscan.core import Application
from operators.isaac_holoscan_bridge.AsyncDataPushOp import AsyncDataPushOp
from operators.isaac_holoscan_bridge.CallbackOp import CallbackOp
from holoscan.operators import HolovizOp
from typing import Callable


class TransmitterApp(Application):
    """A Holoscan application for transmitting simulateddata.

    This application sets up a data transmission pipeline that displays the data locally
    using Holoviz.

    The application consists of two main components:
    1. An AsyncDataPushOp that handles asynchronous data input
    2. The transmitter operator (HolovizOp) that shows the data in a window

    Example:
        >>> # Configure visualization
        >>> viz_spec = HolovizOp.InputSpec(
        ...     name="output",
        ...     type="color",
        ...     width=1920,
        ...     height=1080
        ... )
    """

    def __init__(
        self,
        output_spec: dict,
    ):
        """Initialize the TransmitterApp with output specifications.

        Args:
            output_spec (dict): Dictionary containing the output specifications for the application.
                Each key represents an output name, and the corresponding value should be:
                - A HolovizOp.InputSpec instance for local visualization

        Example:
            >>> output_spec = {
            ...     "viz_output": HolovizOp.InputSpec(...)
            ... }
            >>> app = TransmitterApp(output_spec)
        """
        self._output_spec = output_spec
        self._ready_condition = threading.Condition()
        self._ready = False

        super().__init__()

    def compose(self):
        """Compose the application workflow.

        Sets up the data transmission pipeline by creating and connecting the necessary operators.
        The method creates:
        1. An AsyncDataPushOp for handling input data
        2. The transmitter operator based on the output specifications:
           - HolovizOp for local visualization when using HolovizOp.InputSpec

        The operators are connected in a pipeline where:
        AsyncDataPushOp -> TransmitterOp
        """
        self._async_data_push = AsyncDataPushOp(
            self,
            outputs=self._output_spec.keys(),
            name="Async Data Push",
        )

        # connect the async data push to the transmitters
        for output_name, output_spec in self._output_spec.items():
            if isinstance(output_spec, HolovizOp.InputSpec):
                holoviz = HolovizOp(
                    self,
                    name=f"Holoviz {output_name}",
                    window_title=f"Holoviz {output_name}",
                    tensors=[
                        output_spec,
                    ],
                )
                transmitter = holoviz
                port = "receivers"
            elif isinstance(output_spec, Callable):
                callback = CallbackOp(
                    self,
                    data_ready_callback=output_spec,
                    name=f"Call Function {output_name}",
                )
                transmitter = callback
                port = "in"
            self.add_flow(self._async_data_push, transmitter, {(output_name, port)})

        with self._ready_condition:
            self._ready = True
            self._ready_condition.notify_all()

    def push_data(self, data):
        """Push data into the transmission pipeline.

        This method is used to send data into the application's pipeline. The data will be
        processed by the AsyncDataPushOp and then routed to all configured transmitter
        operators based on the output specifications.

        Args:
            data: The data to be transmitted or displayed. The format of the data should
                match the expected input format of the configured transmitter operators.
        """

        with self._ready_condition:
            while not self._ready:
                self._ready_condition.wait(timeout=1)

        self._async_data_push.push_data(data)