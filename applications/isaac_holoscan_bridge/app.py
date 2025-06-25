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

import argparse
import logging
import os

from Simulator import Simulator
from TransformerApp import TransformerApp


class IsaacSimHoloscan:
    """A class that integrates Isaac Sim with Holoscan for simulation and data processing.

    This class manages the coordination between an Isaac Sim simulator and a Holoscan transformer
    application. It handles the initialization, execution, and cleanup of both components,
    ensuring proper data flow between them.

    Attributes:
        _args: Command line arguments passed to the application
        _futures: List of futures for asynchronous operations
        _simulator: Instance of the Simulator class
        _transformer_app: Instance of the TransformerApp class
    """

    def __init__(self, args):
        """Initialize the IsaacSimHoloscan instance.

        Args:
            args: Command line arguments containing configuration parameters
        """
        self._args = args

        self._futures = []
        self._simulator = None
        self._transformer_app = None

    def run(self):
        """Execute the main simulation and data processing pipeline.

        This method:
        1. Initializes and starts the Isaac Sim simulator
        2. Sets up and runs the Holoscan transformer application
        3. Establishes data flow between simulator and transformer
        4. Handles graceful shutdown of components
        5. Waits for all operations to complete

        The method ensures proper error handling and cleanup of resources.
        """
        logging.info("Starting simulator")

        # start the simulator
        self._simulator = Simulator(
            headless=self._args.headless,
            image_size=(self._args.image_height, self._args.image_width, 4),
            fps=self._args.fps,
            frame_count=self._args.frame_count,
        )

        logging.info("Starting Holoscan pipeline")

        # Set up the Holoscan transformer application
        self._transformer_app = TransformerApp(
            data_ready_callback=self._simulator.data_ready_callback,
            headless=self._args.headless,
            frame_count=self._args.frame_count,
        )

        # Run the transformer application
        transformer_future = self._transformer_app.run_async()

        def transformer_done_callback(future):
            if future.exception():
                logging.fatal(f"TransformerApp failed with exception: {future.exception()}")
                os._exit(1)

        transformer_future.add_done_callback(transformer_done_callback)
        self._futures.append(transformer_future)

        # Run the simulator, this will return if the simulator is finished
        self._simulator.run(self._transformer_app.push_data)

        logging.info("Simulator finished")

        # Shutdown the transformer application
        self._transformer_app.shutdown_async_executor()

        # Wait for the applications to finish
        for future in self._futures:
            future.result()

        logging.info("Transformer application finished")


def main():
    """
    Main entry point for the Isaac Sim Holoscan application.

    This function orchestrates the setup and execution of a simulatied environment that integrates
    Isaac Sim with Holoscan applications. The simulation consists of three main components:
    1. Isaac Sim environment for physics and rendering
    2. Holoscan transformer application for data streaming

    Command Line Arguments:
        --headless (bool): Run Isaac Sim without GUI (default: False)
        --fps (float): FPS of the applications (default: unlimited)

    The function:
    1. Parses command line arguments
    2. Initializes the simulation environment with specified image dimensions (1080x1920x4)
    3. Sets up and runs the transformer application
    4. Executes the simulator with appropriate data push callback
    5. Waits for the transformer application to complete
    6. Handles application failures with appropriate error reporting

    Returns:
        None

    Raises:
        SystemExit: If either transformer or receiver application fails
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim in headless mode",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
        help="Width of the camera image",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
        help="Height of the camera image",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS of the applications",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=-1,
        help="Number of frames to run the application",
    )
    args = parser.parse_args()

    ish = IsaacSimHoloscan(args)
    ish.run()


if __name__ == "__main__":
    main()
