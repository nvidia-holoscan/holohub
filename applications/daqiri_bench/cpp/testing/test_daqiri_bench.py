#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import pytest
from benchmark_utils import parse_benchmark_results
from nvidia_nic_utils import get_nvidia_nics, print_nvidia_nics
from process_utils import run_command
from yaml_config_utils import update_yaml_file

# Configure the logger
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def nvidia_nics():
    """Get NVIDIA NICs and check if we have enough for the tests."""
    nics = get_nvidia_nics()

    if len(nics) < 2:
        logger.warning(f"Not enough NVIDIA NICs available (need at least 2, found {len(nics)})")
        pytest.skip("Not enough NVIDIA NICs available (need at least 2)")

    print_nvidia_nics(nics)
    return nics


@pytest.fixture(scope="module")
def executable(work_dir):
    """Get the path to the daqiri_bench executable."""
    return os.path.join(work_dir, "daqiri_bench")


def skip_if_manager_unavailable(manager):
    """Skip the current test if the given manager is not in DAQIRI_MANAGER_LIST."""
    manager_list = os.environ.get("DAQIRI_MANAGER_LIST", "").split()
    if manager not in manager_list:
        pytest.skip(f"{manager} manager not available in this build")


@pytest.fixture(autouse=True)
def _skip_unavailable_manager(request):
    """Auto-skip parametrized tests whose manager is not in DAQIRI_MANAGER_LIST."""
    if hasattr(request.node, "callspec") and "manager" in request.node.callspec.params:
        skip_if_manager_unavailable(request.node.callspec.params["manager"])


@pytest.mark.parametrize(
    "packet_size,avg_throughput_threshold,missed_pkts_threshold,error_pkts_threshold",
    [
        (64, 6.0, 0.1, 0.0),
        (512, 55.0, 0.1, 0.0),
        (1500, 94.0, 0.1, 0.0),
        (9000, 96.0, 0.1, 0.0),
    ],
)
@pytest.mark.parametrize("manager", ["dpdk"])
def test_multi_if_loopback(
    executable,
    work_dir,
    nvidia_nics,
    manager,
    packet_size,
    avg_throughput_threshold,
    missed_pkts_threshold,
    error_pkts_threshold,
):
    """
    Test 1: TX/RX loopback over single link with one TX queue and one RX queue.

    This test focuses on:
    - Missed packets staying below threshold
    - Errored packets staying below threshold
    - Average throughput staying above threshold
    """
    # Get the first two NICs for this test
    tx_interface, rx_interface = nvidia_nics[0], nvidia_nics[1]

    # Prepare config
    header_size = 64  # Eth (14) + IP (20) + UDP (8) + custom header (22) as defined in yaml config
    payload_size = packet_size - header_size
    in_config_file = os.path.join(work_dir, "daqiri_bench_default_tx_rx.yaml")
    out_config_file = os.path.join(
        work_dir, "testing", f"daqiri_bench_{manager}_tx_rx_{packet_size}.yaml"
    )
    update_yaml_file(
        in_config_file,
        out_config_file,
        {
            "scheduler.max_duration_ms": 10000,
            "daqiri.cfg.interfaces[0].address": tx_interface.bus_id,
            "daqiri.cfg.interfaces[1].address": rx_interface.bus_id,
            "bench_tx.eth_dst_addr": rx_interface.mac_address,
            "bench_tx.payload_size": payload_size,
            "bench_rx.max_packet_size": packet_size,
            "daqiri.cfg.memory_regions[0].buf_size": packet_size,
            "daqiri.cfg.memory_regions[1].buf_size": packet_size,
        },
    )

    # Run the application until completion and parse the results
    command = f"{executable} {out_config_file}"
    result = run_command(command, stream_output=True)
    results = parse_benchmark_results(result.stdout + result.stderr, manager)

    # Validate some expected metrics
    port_map = {0: 1}  # Port 0 (TX) sends to Port 1 (RX), matching daqiri.cfg.interfaces
    missed_pkts_check = results.validate_missed_packets(port_map, missed_pkts_threshold)
    errored_pkts_check = results.validate_errored_packets(port_map, error_pkts_threshold)
    throughput_check = results.validate_throughput(port_map, avg_throughput_threshold)
    assert missed_pkts_check and errored_pkts_check and throughput_check, "Validation failed"
