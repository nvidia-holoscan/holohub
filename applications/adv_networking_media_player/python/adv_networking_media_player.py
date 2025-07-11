#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import sys
from pathlib import Path

from holoscan.core import Application
from holoscan.operators.holoviz import HolovizOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from holoscan.schedulers import MultiThreadScheduler

from holohub.advanced_network_common import _advanced_network_common as adv_network_common
from holohub.advanced_network_media_rx import _advanced_network_media_rx as adv_network_media_rx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_rx_tx_enabled(app, require_rx=True, require_tx=False):
    """
    Check if RX and TX are enabled in the advanced network configuration.

    Args:
        app: The Holoscan Application instance
        require_rx: Whether RX must be enabled (default: True)
        require_tx: Whether TX must be enabled (default: False)

    Returns:
        tuple: (rx_enabled, tx_enabled)

    Raises:
        SystemExit: If required functionality is not enabled
    """
    try:
        adv_net_config_dict = app.kwargs("advanced_network")

        rx_enabled = False
        tx_enabled = False

        # Check if there are interfaces with RX/TX configurations
        if "cfg" in adv_net_config_dict and "interfaces" in adv_net_config_dict["cfg"]:
            for interface in adv_net_config_dict["cfg"]["interfaces"]:
                if "rx" in interface:
                    rx_enabled = True
                if "tx" in interface:
                    tx_enabled = True

        logger.info(f"RX enabled: {rx_enabled}, TX enabled: {tx_enabled}")

        if require_rx and not rx_enabled:
            logger.error("RX is not enabled. Please enable RX in the config file.")
            sys.exit(1)

        if require_tx and not tx_enabled:
            logger.error("TX is not enabled. Please enable TX in the config file.")
            sys.exit(1)

        return rx_enabled, tx_enabled

    except Exception as e:
        logger.warning(f"Could not check RX/TX status from advanced_network config: {e}")
        # Fallback: check if we have the required operator configs
        try:
            if require_rx:
                app.from_config("advanced_network_media_rx")
                logger.info("RX is enabled (found advanced_network_media_rx config)")
            if require_tx:
                app.from_config("advanced_network_media_tx")
                logger.info("TX is enabled (found advanced_network_media_tx config)")
            return require_rx, require_tx
        except Exception as e2:
            if require_rx:
                logger.error("RX is not enabled. Please enable RX in the config file.")
                logger.error(f"Could not find advanced_network_media_rx configuration: {e2}")
                sys.exit(1)
            if require_tx:
                logger.error("TX is not enabled. Please enable TX in the config file.")
                logger.error(f"Could not find advanced_network_media_tx configuration: {e2}")
                sys.exit(1)
            return False, False


class App(Application):
    def compose(self):
        # Initialize advanced network
        try:
            adv_net_config = self.from_config("advanced_network")
            if adv_network_common.adv_net_init(adv_net_config) != adv_network_common.Status.SUCCESS:
                logger.error("Failed to configure the Advanced Network manager")
                sys.exit(1)
            logger.info("Configured the Advanced Network manager")
        except Exception as e:
            logger.error(f"Failed to get advanced network config or initialize: {e}")
            sys.exit(1)

        # Get manager type
        try:
            mgr_type = adv_network_common.get_manager_type()
            logger.info(
                f"Using Advanced Network manager {adv_network_common.manager_type_to_string(mgr_type)}"
            )
        except Exception as e:
            logger.warning(f"Could not get manager type: {e}")

        # Check RX/TX enabled status (require RX for media player)
        check_rx_tx_enabled(self, require_rx=True, require_tx=False)
        logger.info("RX is enabled, proceeding with application setup")

        allocator = UnboundedAllocator(self, name="allocator")

        # Create shared CUDA stream pool for format converters and CUDA operations
        # Optimized sizing for video processing workloads
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        try:
            rx_config = self.kwargs("advanced_network_media_rx")

            adv_net_media_rx = adv_network_media_rx.AdvNetworkMediaRxOp(
                fragment=self,
                **rx_config,
                name="advanced_network_media_rx",
            )

        except Exception as e:
            logger.error(f"Failed to create AdvNetworkMediaRxOp: {e}")
            sys.exit(1)

        # Set up visualization pipeline
        try:
            # Create visualizer
            holoviz_config = self.kwargs("holoviz")
            visualizer = HolovizOp(
                fragment=self,
                name="visualizer",
                allocator=allocator,
                cuda_stream_pool=cuda_stream_pool,
                **holoviz_config,
            )

            self.add_flow(adv_net_media_rx, visualizer, {("out_video_buffer", "receivers")})

        except Exception as e:
            logger.error(f"Failed to set up visualization pipeline: {e}")
            sys.exit(1)

        # Set up scheduler
        try:
            scheduler_config = self.kwargs("scheduler")
            scheduler = MultiThreadScheduler(
                fragment=self, name="multithread-scheduler", **scheduler_config
            )
            self.scheduler(scheduler)
        except Exception as e:
            logger.error(f"Failed to set up scheduler: {e}")
            sys.exit(1)

        logger.info("Application composition completed successfully")


def main():
    if len(sys.argv) < 2:
        logger.error(f"Usage: {sys.argv[0]} config_file")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    # Convert to absolute path if relative
    if not config_path.is_absolute():
        # Get the directory of the script and make path relative to it
        script_dir = Path(sys.argv[0]).parent.resolve()
        config_path = script_dir / config_path

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Using config file: {config_path}")

    try:
        app = App()
        app.config(str(config_path))

        logger.info("Starting application...")
        app.run()

        logger.info("Application finished")

    except Exception as e:
        logger.error(f"Application failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Shutdown advanced network
        try:
            adv_network_common.shutdown()
            logger.info("Advanced Network shutdown completed")
        except Exception as e:
            logger.warning(f"Error during advanced network shutdown: {e}")


if __name__ == "__main__":
    main()
