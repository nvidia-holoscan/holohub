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
import asyncio
import logging
import os
import sys
from pathlib import Path

from app_edge_single_fragment import AppEdgeSingleFragment
from holoscan.core import Tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Endoscopy tool tracking application.")
    parser.add_argument("-d", "--data", type=str, help="Path to the data directory")
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    return args.data, args.config


async def main():
    data_directory, config_path = parse_arguments()

    if not data_directory:
        data_directory = os.getenv("HOLOSCAN_INPUT_PATH")
        if not data_directory or not os.path.isdir(data_directory):
            data_directory = Path.cwd() / "data" / "endoscopy"
            if not data_directory.is_dir():
                logger.error(
                    "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable."
                )
                sys.exit(-1)

    if not config_path:
        config_path = os.getenv("HOLOSCAN_CONFIG_PATH")
        if not config_path:
            config_path = Path(sys.argv[0]).parent.parent / "endoscopy_tool_tracking.yaml"

    app = AppEdgeSingleFragment(data_directory)
    app.config(str(config_path))

    try:
        with Tracker(app) as trackers:
            future = app.run_async()
            await app.start_streaming_client()

            future.result()
            trackers.print()
    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
