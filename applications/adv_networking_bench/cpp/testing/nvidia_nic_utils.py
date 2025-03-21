#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class NetworkInterface:
    """Class representing a network interface with its properties."""
    interface_name: str   # Interface name (e.g., "ens3" or "cx7_0")
    bus_id: str           # PCI bus ID (e.g., "0000:3b:00.0")
    mac_address: str      # MAC address
    is_up: bool           # Whether the interface is up

    def __str__(self) -> str:
        status = "Up" if self.is_up else "Down"
        return f"NetworkInterface(name={self.interface_name}, bus_id={self.bus_id}, mac={self.mac_address}, status={status})"


def run_command(cmd: str) -> Tuple[int, str, str]:
    """
    Run a shell command and return the result.

    Args:
        cmd: Command to run

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def get_nvidia_nics() -> List[NetworkInterface]:
    """
    Get a list of NVIDIA NICs on the system using ibdev2netdev.

    Returns:
        List of NetworkInterface objects representing NVIDIA NICs
    """
    # Use ibdev2netdev to get NVIDIA/Mellanox NIC information
    returncode, stdout, stderr = run_command("ibdev2netdev -v 2>/dev/null")
    if returncode != 0:
        logger.error(f"Failed to get NVIDIA NICs with ibdev2netdev: {stderr}")
        return []

    # Parse the output of ibdev2netdev -v
    # Example output:
    # 0005:03:00.0 mlx5_0 (MT4129 -            )                 fw 28.39.3004 port 1 (ACTIVE) ==> cx7_0 (Up)
    # 0005:03:00.1 mlx5_1 (MT4129 -            )                 fw 28.39.3004 port 1 (ACTIVE) ==> cx7_1 (Up)
    interfaces = []
    for line in stdout.strip().split('\n'):
        if not line:
            continue

        # Parse bus ID and interface name
        bus_id_match = re.match(r'^([0-9a-f:.]+)\s+', line)
        if_match = re.search(r'==>\s+(\w+)\s+\((\w+)\)', line)

        if not bus_id_match or not if_match:
            logger.debug(f"Could not parse interface info from line: {line}")
            continue

        bus_id = bus_id_match.group(1)
        interface_name = if_match.group(1)
        status = if_match.group(2)
        is_up = status.lower() == "up"

        # Get MAC address
        returncode, mac_stdout, mac_stderr = run_command(f"cat /sys/class/net/{interface_name}/address")
        if returncode != 0:
            logger.warning(f"Failed to get MAC address for interface {interface_name}: {mac_stderr}")
            continue

        mac_address = mac_stdout.strip()

        # Create and add the interface
        interface = NetworkInterface(
            interface_name=interface_name,
            bus_id=bus_id,
            mac_address=mac_address,
            is_up=is_up
        )

        interfaces.append(interface)

    return interfaces


def print_nvidia_nics(nics: List[NetworkInterface]):
    """
    Print a list of NVIDIA NIC interfaces.
    """
    up_nics = [nic for nic in nics if nic.is_up]
    logger.info(f"Found {len(nics)} NVIDIA NIC interfaces ({len(up_nics)} are UP)")
    for nic in nics:
        logger.info(nic)


if __name__ == "__main__":
    # Set up console logging if run directly
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    print_nvidia_nics(get_nvidia_nics())
