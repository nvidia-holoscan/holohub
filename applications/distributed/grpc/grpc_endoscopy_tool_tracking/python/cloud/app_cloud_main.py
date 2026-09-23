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

import asyncio
import logging
import os
import signal
import sys
from argparse import ArgumentParser
from pathlib import Path
from queue import Queue

from endoscopy_tool_tracking import EndoscopyToolTrackingPipeline

from operators.grpc_operators.python.server.application_factory import (
    ApplicationFactory,
    ApplicationInstance,
)
from operators.grpc_operators.python.server.entity_servicer import HoloscanEntityServicer
from operators.grpc_operators.python.server.grpc_service import GrpcService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def signal_handler():
    logger.warning("Stopping services...")
    grpc_service = GrpcService()
    await grpc_service.stop()


def _create_application_instance(args, request_queue: Queue, response_queue: Queue):
    instance = ApplicationInstance(EndoscopyToolTrackingPipeline(request_queue, response_queue))
    instance.instance.config(str(args.config))
    instance.instance.data_path = args.data
    instance.start_application()
    return instance


def parse_arguments():
    parser = ArgumentParser(description="gRPC-enabled Endoscopy tool tracking demo application.")

    parser.add_argument(
        "-c",
        "--config",
        default=os.environ.get(
            "HOLOSCAN_CONFIG_PATH",
            Path(sys.argv[0]).parent.parent / "endoscopy_tool_tracking.yaml",
        ),
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default=os.environ.get("HOLOSCAN_INPUT_PATH", f"{os.getcwd()}/data/endoscopy"),
        help=("Set the data path (default: %(default)s)."),
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=50051,
        help=("Set the gRPC Server listening port  (default: %(default)s)."),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind IP address or localhost; non-loopback requires mutual TLS",
    )
    parser.add_argument("--tls-cert", type=Path, help="PEM server certificate chain")
    parser.add_argument("--tls-key", type=Path, help="PEM server private key")
    parser.add_argument(
        "--tls-ca", type=Path, help="PEM CA certificates trusted for client authentication"
    )
    args = parser.parse_args()
    tls = (args.tls_cert, args.tls_key, args.tls_ca)
    if any(tls) and not all(tls):
        parser.error("--tls-cert, --tls-key, and --tls-ca must be provided together")
    return args


async def main(loop):
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(signal_handler()))

    args = parse_arguments()

    if not os.path.isdir(args.data):
        raise ValueError(
            f"Data path '{args.data}' does not exist. Use --data or set HOLOSCAN_INPUT_PATH environment variable."
        )

    application_factory = ApplicationFactory()
    application_factory.register_application(
        "EndoscopyToolTracking",
        lambda request_queue, response_queue: _create_application_instance(
            args, request_queue, response_queue
        ),
    )

    # Initialize the gRPC service
    grpc_service = GrpcService()
    grpc_service.initialize(
        args.port,
        application_factory,
        host=args.host,
        private_key=args.tls_key.read_bytes() if args.tls_key else None,
        certificate_chain=args.tls_cert.read_bytes() if args.tls_cert else None,
        root_certificates=args.tls_ca.read_bytes() if args.tls_ca else None,
    )

    # Configure the gRPC services
    servicer = HoloscanEntityServicer("EndoscopyToolTracking")

    # Start the gRPC server and services
    await grpc_service.start([servicer])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
