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

from typing import Callable

import holoscan
from operators.isaac_holoscan_bridge.CallbackOp import CallbackOp
from operators.isaac_holoscan_bridge.DataProviderOp import DataProviderOp


class ReceiverApp(holoscan.core.Application):
    """A Holoscan application for receiving data from local data generation.

    The application uses CUDA for GPU-accelerated data processing and supports configurable
    buffer sizes and frame rates.
    """

    def __init__(
        self,
        fps: float,
        data_ready_callback: Callable,
    ):
        """Initialize the ReceiverApp.

        Args:
            fps (float): Frames per second for data generation in local mode
            data_ready_callback (Callable): Callback function to be called when new data is available
        """
        self._fps = fps
        self._data_ready_callback = data_ready_callback
        super().__init__()

    def compose(self):
        """Compose the application workflow by setting up the data pipeline.

        This method configures either:
        1. A network receiver (RoCE or Linux) if hololink_ip is provided
        2. A local data provider if no network interface is specified

        The method also initializes CUDA context for GPU operations and sets up
        the necessary data channels and operators for data flow.
        """
        # Generate synthetic data
        source = DataProviderOp(
            self,
            buffer_size=self._buffer_size,
            fps=self._fps,
            name="Data Provider",
        )
        port = "out"

        self._callback = CallbackOp(
            self,
            data_ready_callback=self._data_ready_callback,
            name="Callback",
        )

        self.add_flow(source, self._callback, {(port, "in")})
