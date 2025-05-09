#!/usr/bin/env python3
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

import logging
import os
from threading import Thread
from time import sleep

import pytest
from benchmark_utils import parse_benchmark_results
from nvidia_nic_utils import get_nvidia_nics, print_nvidia_nics
from process_utils import monitor_process, run_command, start_process
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
    """Get the path to the adv_networking_bench executable."""
    return os.path.join(work_dir, "adv_networking_bench")


@pytest.mark.parametrize(
    "packet_size,avg_throughput_threshold,missed_pkts_threshold,error_pkts_threshold",
    [
        (64, 6.0, 0.1, 0.0),
        (512, 55.0, 0.1, 0.0),
        (1500, 94.0, 0.1, 0.0),
        (9000, 96.0, 0.1, 0.0),
    ],
)
@pytest.mark.parametrize("manager", ["dpdk", "gpunetio"])
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
    in_config_file = os.path.join(work_dir, "adv_networking_bench_default_tx_rx.yaml")
    out_config_file = os.path.join(
        work_dir, "testing", f"adv_networking_bench_{manager}_tx_rx_{packet_size}.yaml"
    )
    update_yaml_file(
        in_config_file,
        out_config_file,
        {
            "advanced_network.cfg.manager": manager,
            "scheduler.max_duration_ms": 10000,
            "advanced_network.cfg.interfaces[0].address": tx_interface.bus_id,
            "advanced_network.cfg.interfaces[1].address": rx_interface.bus_id,
            "bench_tx.eth_dst_addr": rx_interface.mac_address,
            "bench_tx.payload_size": payload_size,
            "bench_rx.max_packet_size": packet_size,
            "advanced_network.cfg.memory_regions[0].buf_size": packet_size,
            "advanced_network.cfg.memory_regions[1].buf_size": packet_size,
        },
    )

    # Run the application until completion and parse the results
    command = f"{executable} {out_config_file}"
    result = run_command(command, stream_output=True)
    results = parse_benchmark_results(result.stdout + result.stderr, manager)

    # Validate some expected metrics
    port_map = {0: 1}  # Port 0 (TX) sends to Port 1 (RX), match advanced_network.cfg.interfaces
    missed_pkts_check = results.validate_missed_packets(port_map, missed_pkts_threshold)
    errored_pkts_check = results.validate_errored_packets(port_map, error_pkts_threshold)
    throughput_check = results.validate_throughput(port_map, avg_throughput_threshold)
    assert missed_pkts_check and errored_pkts_check and throughput_check, "Validation failed"


def test_multi_rx_q(executable, work_dir, nvidia_nics):
    """
    Test 2: RX multi queue with a single CPU core using scapy to send packets.

    This test focuses on:
    - Validating the distribution of packets to RX queues
    """
    # Get the NICs for this test
    tx_interface, rx_interface = nvidia_nics[0], nvidia_nics[1]

    # Prepare config
    config_file = os.path.join(work_dir, "adv_networking_bench_default_rx_multi_q.yaml")
    update_yaml_file(
        config_file,
        config_file,
        {
            "scheduler.max_duration_ms": 10000,
            "advanced_network.cfg.interfaces[0].address": rx_interface.bus_id,
        },
    )

    # Run the application (non-blocking)
    command = f"{executable} {config_file}"
    p = start_process(command)

    # Send packets with scapy after a 5s delay
    def send_test_packets():
        logger.info("Sleeping before sending test packets")
        sleep(5)
        try:
            from scapy.all import IP, UDP, Ether, sendp

            logger.info(
                f"Sending test packets to queues using interface {tx_interface.interface_name}"
            )
            packet1 = (
                Ether() / IP(dst="foo") / UDP(sport=4095, dport=4095) / ("X" * (1050 - 20 - 8))
            )
            packet2 = (
                Ether() / IP(dst="foo") / UDP(sport=4096, dport=4096) / ("X" * (1050 - 20 - 8))
            )

            # Send one packet to each queue
            sendp(packet1, iface=tx_interface.interface_name, count=1, verbose=1)
            sendp(packet2, iface=tx_interface.interface_name, count=1, verbose=1)
            logger.info("Test packets sent successfully")
        except Exception as e:
            logger.error(f"Failed to send test packets: {e}")
            raise

    # Start the packet sending thread (non-blocking)
    Thread(target=send_test_packets).start()

    # Monitor the application until completion and parse the results
    result = monitor_process(p)
    results = parse_benchmark_results(result.stdout + result.stderr, "dpdk")

    # For this test, we only care about queue packet distribution (on port 0)
    expected_q_pkts = {0: 1, 1: 1}  # Expecting 1 packet for both queue 0 and 1
    queue_check = results.validate_rx_queue_packets(0, expected_q_pkts)  # On port 0
    assert queue_check, "Queue packet distribution validation failed"


def test_hds_rx(executable, work_dir, nvidia_nics):
    """
    Test 3: RX with header-data split.
    """
    # Get the first two NICs for this test
    tx_interface, rx_interface = nvidia_nics[0], nvidia_nics[1]

    # Prepare config
    config_file = os.path.join(work_dir, "adv_networking_bench_default_tx_rx_hds.yaml")
    update_yaml_file(
        config_file,
        config_file,
        {
            "scheduler.max_duration_ms": 10000,
            "advanced_network.cfg.interfaces[0].address": tx_interface.bus_id,
            "advanced_network.cfg.interfaces[1].address": rx_interface.bus_id,
            "bench_tx.eth_dst_addr": rx_interface.mac_address,
        },
    )

    # Run the application until completion and parse the results
    command = f"{executable} {config_file}"
    result = run_command(command, stream_output=True)
    results = parse_benchmark_results(result.stdout + result.stderr, "dpdk")

    # Validate some expected metrics
    port_map = {0: 1}  # Port 0 (TX) sends to Port 1 (RX)
    avg_throughput_threshold = 85.0
    missed_pkts_threshold = 0.1
    error_pkts_threshold = 0.0
    missed_pkts_check = results.validate_missed_packets(port_map, missed_pkts_threshold)
    errored_pkts_check = results.validate_errored_packets(port_map, error_pkts_threshold)
    throughput_check = results.validate_throughput(port_map, avg_throughput_threshold)
    assert missed_pkts_check and errored_pkts_check and throughput_check, "Validation failed"


def test_gpunetio_single_if_loopback(executable, work_dir, nvidia_nics):
    """
    Test 4: GPUNetIO with single interface loopback.
    """
    # Get the first two NICs for this test
    interface = nvidia_nics[0]

    # Prepare config
    config_file = os.path.join(work_dir, "adv_networking_bench_gpunetio_tx_rx.yaml")
    update_yaml_file(
        config_file,
        config_file,
        {
            "scheduler.max_duration_ms": 10000,
            "advanced_network.cfg.interfaces[0].address": interface.bus_id,
            "bench_tx.eth_dst_addr": interface.mac_address,
            "bench_tx.address": interface.bus_id,
        },
    )

    # Run the application until completion
    command = f"{executable} {config_file}"
    result = run_command(command, stream_output=True)
    parse_benchmark_results(result.stdout + result.stderr, "gpunetio")
    assert result.returncode == 0, "Application errored out"
    assert "[error]" not in (result.stdout + result.stderr), "Application reported errors"
    assert "[ERR]" not in (result.stdout + result.stderr), "Application reported errors"


def test_multi_q_hds_tx_rx(executable, work_dir, nvidia_nics):
    """
    Test 5: RX with multi-queue and header-data split.
    """
    # Get the first two NICs for this test
    tx_interface, rx_interface = nvidia_nics[0], nvidia_nics[1]

    # Prepare config
    config_file = os.path.join(work_dir, "adv_networking_bench_default_tx_rx_multi_q_hds.yaml")
    update_yaml_file(
        config_file,
        config_file,
        {
            "scheduler.max_duration_ms": 10000,
            "advanced_network.cfg.interfaces[0].address": tx_interface.bus_id,
            "advanced_network.cfg.interfaces[1].address": rx_interface.bus_id,
            "bench_tx.eth_dst_addr": rx_interface.mac_address,
            "advanced_network.cfg.tx_meta_buffers": 4096,
            "advanced_network.cfg.rx_meta_buffers": 4096,
        },
    )

    # Run the application until completion and parse the results
    command = f"{executable} {config_file}"
    result = run_command(command, stream_output=True)
    results = parse_benchmark_results(result.stdout + result.stderr, "dpdk")

    # Validate some expected metrics
    # We only check that every queue got at least 100 packets here
    rx_queue_pkts_check = results.validate_rx_queue_packets(
        1, {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100, 7: 100, 8: 100}, gt=True
    )
    assert rx_queue_pkts_check, "RX queue packet distribution validation failed"
