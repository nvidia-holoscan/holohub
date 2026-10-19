# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import grpc
from app_edge_single_fragment import AppEdgeSingleFragment
from holoscan.core import Tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Endoscopy tool tracking application.")
    parser.add_argument("-d", "--data", type=str, help="Path to the data directory")
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file")
    parser.add_argument("--tls-cert", type=Path, help="PEM client certificate chain")
    parser.add_argument("--tls-key", type=Path, help="PEM client private key")
    parser.add_argument(
        "--tls-ca", type=Path, help="PEM CA certificates trusted for server authentication"
    )
    args = parser.parse_args()
    tls = (args.tls_cert, args.tls_key, args.tls_ca)
    if any(tls) and not all(tls):
        parser.error("--tls-cert, --tls-key, and --tls-ca must be provided together")
    return args


async def main():
    args = parse_arguments()
    data_directory, config_path = args.data, args.config

    credentials = None
    if args.tls_ca:
        ca, key, certificate = (
            path.read_bytes() for path in (args.tls_ca, args.tls_key, args.tls_cert)
        )
        if not all((ca, key, certificate)):
            raise ValueError("Mutual TLS credential files must not be empty")
        credentials = grpc.ssl_channel_credentials(ca, key, certificate)

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

    app = AppEdgeSingleFragment(data_directory, credentials=credentials)
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
