# SPDX-FileCopyrightText: Copyright (c) 2022-2025 DELTACAST.TV. All rights reserved.
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
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import FormatConverterOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

from holohub.videomaster import VideoMasterTransmitterOp


class DeltacastTransmitterApp(Application):
    def __init__(self, data_path):
        """Initialize the deltacast transmitter application.
        Parameters
        ----------
            data_path (str): Path to the data directory. The data will be transmit on the deltacast card.
        """
        super().__init__()
        self.name = "Deltacast Transmitter"
        self.data_path = data_path

    def compose(self):
        """
        Compose the application by setting up the operators and their connections.
        """
        # Retrieve VideoMaster parameters
        videomaster_kwargs = self.kwargs("videomaster")
        width = videomaster_kwargs.get("width", 1920)
        height = videomaster_kwargs.get("height", 1080)

        # Calculate source block size and count and define the source pool parameters
        source_block_size = width * height * 4 * 4
        source_block_count = 3 if videomaster_kwargs.get("rdma") else 4

        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=source_block_size,
            num_blocks=source_block_count,
        )

        # Initialize operators
        source = VideoStreamReplayerOp(
            self, name="replayer", directory=self.data_path, **self.kwargs("replayer")
        )

        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
            **self.kwargs("output_format_converter"),
        )

        visualizer = VideoMasterTransmitterOp(
            self,
            name="videomaster",
            pool=UnboundedAllocator(self, name="pool"),
            rdma=videomaster_kwargs.get("rdma", False),
            board=videomaster_kwargs.get("board", 0),
            width=width,
            height=height,
            output=videomaster_kwargs.get("output", 0),
            progressive=videomaster_kwargs.get("progressive", True),
            framerate=videomaster_kwargs.get("framerate", 60),
            enable_overlay=videomaster_kwargs.get("enable_overlay", False),
        )

        # Define the data flow between operators
        self.add_flow(source, format_converter)
        self.add_flow(format_converter, visualizer)


def parse_config():
    """
    Parse command line arguments and validate paths.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Default data path
    default_data_path = os.path.join(os.getcwd(), "data/endoscopy")

    # Parse command line arguments
    parser = ArgumentParser(description="DeltaCast Transmitter demo application.")
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the data directory. (default location:  %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "deltacast_transmitter.yaml"),
        help="Path to the configuration file to override the default config file location. If not provided, the deltacast_transmitter.yaml in root directory will be used. (default location:  %(default)s)",
    )

    args = parser.parse_args()

    # Ensure the data path exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data path {args.data_path} does not exist. Use --data_path to specify the correct path."
        )

    # Ensure the configuration file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Configuration file {args.config} does not exist at expected location. Use --config to specify the correct path."
        )

    return args


def main():
    try:
        args = parse_config()
        app = DeltacastTransmitterApp(data_path=args.data_path)
        app.config(args.config)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
