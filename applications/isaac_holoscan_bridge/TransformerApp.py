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
from typing import Callable

from holoscan.core import Application
from holoscan.operators import HolovizOp
from operators.isaac_holoscan_bridge.AsyncDataPushOp import AsyncDataPushOp
from operators.isaac_holoscan_bridge.CallbackOp import CallbackOp
from GaussianBlurOp import GaussianBlurOp


class TransformerApp(Application):
    """A Holoscan application that processes and visualizes camera data from Isaac Sim.

    This application creates a pipeline that:
    1. Receives camera image and pose data asynchronously
    2. Applies Gaussian blur to the camera image
    3. Visualizes the camera image using Holoviz
    4. Provides the processed data through a callback mechanism

    The application uses multiple operators including AsyncDataPushOp, GaussianBlurOp,
    HolovizOp, and CallbackOp to create a complete data processing pipeline.
    """

    def __init__(
        self,
        data_ready_callback: Callable,
    ):
        """Initialize the TransformerApp.

        Args:
            data_ready_callback (Callable): A callback function that will be called when
                new data is ready for processing. This function should accept camera image
                and pose data as arguments.
        """
        self._ready_condition = threading.Condition()
        self._ready = False

        self._data_ready_callback = data_ready_callback

        super().__init__()

    def compose(self):
        """Compose the application by creating and connecting the operators.

        This method sets up the processing pipeline by:
        1. Creating all necessary operators
        2. Configuring their parameters
        3. Establishing the data flow between operators
        4. Signaling that the application is ready to receive data
        """
        self._async_data_push_op = AsyncDataPushOp(
            self,
            outputs=["camera_image", "camera_pose"],
            name="Async Data Push",
        )

        self._holoviz_op = HolovizOp(
            self,
            name="Holoviz camera_image",
            window_title="Isaac Sim Holoscan Camera Image",
            width=810,
            height=540,
        )

        self._gaussian_blur_op = GaussianBlurOp(
            self,
            name="Gaussian Blur",
        )
        self._callback_op = CallbackOp(
            self,
            data_ready_callback=self._data_ready_callback,
            inputs=["camera_image", "camera_pose"],
            name="Callback",
        )

        self.add_flow(self._async_data_push_op, self._gaussian_blur_op, {("camera_image", "input")})
        self.add_flow(self._async_data_push_op, self._holoviz_op, {("camera_image", "receivers")})
        self.add_flow(self._gaussian_blur_op, self._callback_op, {("output", "camera_image")})
        self.add_flow(self._async_data_push_op, self._callback_op, {("camera_pose", "camera_pose")})

        # signal that the application is ready
        with self._ready_condition:
            self._ready = True
            self._ready_condition.notify_all()

    def push_data(self, data):
        """Push new data into the processing pipeline.

        This method waits for the application to be ready before pushing data
        to the AsyncDataPushOp operator.

        Args:
            data: The data to be pushed into the pipeline. This should contain
                camera image and pose information.
        """
        with self._ready_condition:
            while not self._ready:
                self._ready_condition.wait(timeout=1)

        self._async_data_push_op.push_data(data)
