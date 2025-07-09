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

from AsyncDataPushOp import AsyncDataPushOp
from CallbackOp import CallbackOp
from ControlOp import ControlOp
from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import HolovizOp
from SobelOp import SobelOp


class TransformerApp(Application):
    """A Holoscan application that processes and visualizes camera data from Isaac Sim.

    This application creates a pipeline that:
    1. Receives camera image and arm joint positions data asynchronously
    2. Applies Sobel edge detection to the camera image
    3. Visualizes the camera image using Holoviz
    4. Controls the arm joint positions
    5. Provides the processed data through a callback mechanism

    The application uses multiple operators including AsyncDataPushOp, SobelOp,
    HolovizOp, ControlOp, and CallbackOp to create a complete data processing pipeline.
    """

    def __init__(
        self,
        data_ready_callback: Callable,
        headless: bool,
        frame_count: int,
    ):
        """Initialize the TransformerApp.

        Args:
            data_ready_callback (Callable): A callback function that will be called when
                new data is ready for processing. This function should accept camera image
                and pose data as arguments.
            headless (bool): Whether to run the application in headless mode
            frame_count (int): The number of frames to run the application (-1 for infinite)
        """
        self._ready_condition = threading.Condition()
        self._headless = headless
        self._frame_count = frame_count

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
            outputs=["camera_image", "arm_joint_positions"],
            name="Async Data Push",
        )
        if self._frame_count != -1:
            self._async_data_push_op.add_arg(CountCondition(self, count=self._frame_count))

        # The HolovizOp has two views, one for the camera image and one for the Sobel image.
        # The camera image is displayed in the top half of the window, and the Sobel image is
        # displayed in the bottom half.
        camera_image_view = HolovizOp.InputSpec.View()
        camera_image_view.height = 0.5
        camera_image_sobel_view = HolovizOp.InputSpec.View()
        camera_image_sobel_view.offset_y = 0.5
        camera_image_sobel_view.height = 0.5

        self._holoviz_op = HolovizOp(
            self,
            name="Holoviz",
            window_title="Isaac Sim Holoscan Visualization",
            width=810,
            height=1080,
            headless=self._headless,
            tensors=[
                dict(name="camera_image", type="color", views=[camera_image_view]),
                dict(name="camera_image_sobel", type="color", views=[camera_image_sobel_view]),
            ],
        )

        self._sobel_op = SobelOp(
            self,
            name="Sobel",
        )

        self._control_op = ControlOp(
            self,
            name="Control",
        )

        self._callback_op = CallbackOp(
            self,
            data_ready_callback=self._data_ready_callback,
            inputs=["camera_image"],
            optional_inputs=["arm_joint_positions"],
            name="Callback",
        )

        self.add_flow(self._async_data_push_op, self._sobel_op, {("camera_image", "input")})
        self.add_flow(self._sobel_op, self._callback_op, {("output", "camera_image")})
        self.add_flow(
            self._async_data_push_op,
            self._control_op,
            {("arm_joint_positions", "input_joint_positions")},
        )
        self.add_flow(
            self._control_op, self._callback_op, {("output_joint_positions", "arm_joint_positions")}
        )

        # add the flow for the visualization
        self.add_flow(self._sobel_op, self._holoviz_op, {("output", "receivers")})
        self.add_flow(self._async_data_push_op, self._holoviz_op, {("camera_image", "receivers")})

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
