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
from holoscan.operators import FormatConverterOp, HolovizOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

from holohub.videomaster import VideoMasterSourceOp


class DeltacastReceiverApp(Application):
    def __init__(self):
        """Initialize the deltacast receiver application."""
        super().__init__()
        self.name = "Deltacast Receiver"

    def compose(self):
        """
        Compose the application by setting up the operators and their connections.
        """
        # Retrieve VideoMaster parameters
        deltacast_kwargs = self.kwargs("deltacast")
        width = deltacast_kwargs.get("width", 1920)
        height = deltacast_kwargs.get("height", 1080)
        use_rdma = deltacast_kwargs.get("rdma", False)

        # Calculate source block size and count and define the source pool parameters
        source_block_size = width * height * 4 * 4
        source_block_count = 3 if use_rdma else 4

        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=source_block_size,
            num_blocks=source_block_count,
        )

        # Initialize operators
        source = VideoMasterSourceOp(
            self,
            name="deltacast_source",
            pool=UnboundedAllocator(self, name="source_pool"),
            rdma=use_rdma,
            board=deltacast_kwargs.get("board", 0),
            input=deltacast_kwargs.get("input", 0),
            width=width,
            height=height,
            progressive=deltacast_kwargs.get("progressive", True),
            framerate=deltacast_kwargs.get("framerate", 30),
        )

        # Format converter to prepare for visualization
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=BlockMemoryPool(self, name="converter_pool", **source_pool_kwargs),
            **self.kwargs("format_converter"),
        )

        # Drop alpha channel converter (matches C++ implementation exactly)
        drop_alpha_channel_converter = FormatConverterOp(
            self,
            name="drop_alpha_channel_converter",
            pool=BlockMemoryPool(self, name="drop_alpha_pool", **source_pool_kwargs),
            **self.kwargs("drop_alpha_channel_converter"),
        )

        # Holoviz for visualization
        visualizer = HolovizOp(
            self,
            name="holoviz",
            allocator=UnboundedAllocator(self, name="holoviz_allocator"),
            **self.kwargs("holoviz"),
        )

        # Connect the pipeline: source -> drop_alpha_channel_converter -> format_converter -> holoviz
        # This matches the exact flow from the C++ implementation
        self.add_flow(source, drop_alpha_channel_converter)
        self.add_flow(drop_alpha_channel_converter, format_converter)
        self.add_flow(format_converter, visualizer, {("", "receivers")})


def parse_config():
    """
    Parse command line arguments and validate paths.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # Parse command line arguments
    parser = ArgumentParser(description="DeltaCast Receiver demo application.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "deltacast_receiver.yaml"),
        help="Path to the configuration file to override the default config file location. If not provided, the deltacast_receiver.yaml in root directory will be used. (default location:  %(default)s)",
    )

    args = parser.parse_args()

    # Ensure the configuration file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Configuration file {args.config} does not exist at expected location. Use --config to specify the correct path."
        )

    return args


def main():
    try:
        args = parse_config()
        app = DeltacastReceiverApp()
        app.config(args.config)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()