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

import os
import argparse
import logging

import holoscan


class IsaacSimHoloscan:
    def __init__(self, args, image_size):
        self._args = args
        self._image_size = image_size

        self._futures = []
        self._simulator = None
        self._transmitter_app = None
        self._transformer_app = None

    def run(self):
        if self._args.simulator:
            logging.info("Starting simulator")

            from Simulator import Simulator

            # start the simulator
            self._simulator = Simulator(
                self._args.headless, self._image_size, self._args.fps
            )

        if self._args.transmitter:
            logging.info("Starting transmitter")

            from holoscan.operators import HolovizOp
            from TransmitterApp import TransmitterApp

            # these are the names of the output ports of the TransmitterApp, the simulator will push data to these ports
            output_spec = dict()

            # Set the output of the TransmitterApp to the HolovizOp
            output_spec["camera_image"] = HolovizOp.InputSpec(
                "", HolovizOp.InputType.COLOR
            )

            # print the camera pose
            output_spec["camera_pose"] = lambda data: print(data)

            # Set up the Holoscan transmitter application
            self._transmitter_app = TransmitterApp(output_spec=output_spec)

            # Run the transmitter application
            transmitter_future = self._transmitter_app.run_async()

            def transmitter_done_callback(future):
                if future.exception():
                    logging.fatal(
                        f"TransmitterApp failed with exception: {future.exception()}"
                    )
                    os._exit(1)

            transmitter_future.add_done_callback(transmitter_done_callback)
            self._futures.append(transmitter_future)

        if self._args.receiver:
            logging.info("Starting receiver")

            from ReceiverApp import ReceiverApp

            self._receiver_app = ReceiverApp(
                ibv_name=self._args.ibv_name_rx,
                ibv_port=self._args.ibv_port_rx,
                hololink_ip=self._args.hololink_ip_rx,
                buffer_size=self._image_size[0]
                * self._image_size[1]
                * self._image_size[2],
                fps=self._args.fps,
                data_ready_callback=lambda data: self._simulator.data_ready_callback(
                    data.data
                )
                if self._simulator
                else lambda data: print(data),
            )

            def receiver_done_callback(future):
                if future.exception():
                    print(f"ReceiverApp failed with exception: {future.exception()}")
                    os._exit(1)

            receiver_future = self._receiver_app.run_async()
            receiver_future.add_done_callback(receiver_done_callback)
            self._futures.append(receiver_future)

        if self._args.simulator:
            # Run the simulator, this will return if the simulator is finished
            self._simulator.run(
                self._transmitter_app.push_data
                if self._transmitter_app
                else lambda data: None
            )
        elif self._args.transmitter:
            # no Simulator, just push images to the transmitter
            data = dict()

            import cupy as cp

            green = cp.array([255, 0, 255, 255], dtype=cp.uint8)
            data["camera_image"] = cp.full(self._image_size, green, dtype=cp.uint8)

            data["camera_pose"] = "dummy pose"
            while True:
                self._transmitter_app.push_data(data)

        # Wait for the applications to finish
        for future in self._futures:
            future.result()


def main():
    """
    Main entry point for the IsaacSim Holoscan application.

    This function orchestrates the setup and execution of a simulatied environment that integrates
    IsaacSim with Holoscan applications. The simulation consists of three main components:
    1. IsaacSim environment for physics and rendering
    2. Holoscan transmitter application for data streaming

    Command Line Arguments:
        --log-level (int): Logging verbosity level (default: 20)
        --headless (bool): Run IsaacSim without GUI (default: False)
        --transmitter (bool): Enable transmitter application (default: True)
        --receiver (bool): Enable receiver application (default: True)
        --simulator (bool): Enable simulator (default: True)
        --fps (float): FPS of the applications (default: 10.0)

    The function:
    1. Parses command line arguments
    2. Initializes the simulation environment with specified image dimensions (1080x1920x4)
    3. Configures logging for holoscan
    4. Sets up and runs the transmitter application if enabled:
       - Configures output specifications for camera image and pose
       - Uses HolovizOp for local display
    5. Sets up and runs the receiver application if enabled:
       - Links with transmitter metadata if available
    6. Executes the simulator with appropriate data push callback
    7. Waits for both transmitter and receiver applications to complete
    8. Handles application failures with appropriate error reporting

    Returns:
        None

    Raises:
        SystemExit: If either transmitter or receiver application fails
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--headless",
        type=bool,
        default=False,
        help="Run IsaacSim in headless mode",
    )
    parser.add_argument(
        "--transmitter",
        action="store_true",
        help="Run the transmitter application",
    )
    parser.add_argument(
        "--receiver",
        action="store_true",
        help="Run the receiver application",
    )
    parser.add_argument(
        "--simulator",
        action="store_true",
        help="Run the simulator",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS of the applications",
    )
    args = parser.parse_args()

    # set up logging
    if args.log_level == logging.getLevelName("TRACE"):
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.TRACE)
    elif args.log_level == logging.DEBUG:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.DEBUG)
    elif args.log_level == logging.INFO:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.INFO)
    elif args.log_level == logging.WARNING:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.WARN)
    elif args.log_level == logging.ERROR:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.ERROR)
    elif args.log_level == logging.CRITICAL:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.CRITICAL)
    elif args.log_level == logging.NOTSET:
        holoscan.logger.set_log_level(holoscan.logger.LogLevel.OFF)

    image_size = (1080, 1920, 4)

    ish = IsaacSimHoloscan(args, image_size)
    ish.run()


if __name__ == "__main__":
    main()
