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
from threading import Thread
from time import sleep

import pytest
from benchmark_utils import parse_benchmark_results
from io_utils import generate_smpte2110_file
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
    # Skip if the manager is not available
    manager_list = os.environ.get("ANO_MANAGER_LIST", "").split()
    if manager not in manager_list:
        pytest.skip(f"{manager} manager not available in this build")

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
    # Skip if GPUNetIO manager is not available
    manager_list = os.environ.get("ANO_MANAGER_LIST", "").split()
    if "gpunetio" not in manager_list:
        pytest.skip("gpunetio manager not available in this build")

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
    # We only check that every rx queue got at least 1 packet here on port 1
    expected_q_pkts = {i: 1 for i in range(9)}
    rx_queue_pkts_check = results.validate_rx_queue_packets(1, expected_q_pkts, allow_greater=True)
    assert rx_queue_pkts_check, "RX queue packet distribution validation failed"


@pytest.mark.parametrize("rivermax_receiver_type", ["ipo_receiver", "rtp_receiver"])
def test_rivermax_tx_rx(executable, work_dir, nvidia_nics, rivermax_receiver_type):
    """
    Test 6: Rivermax TX/RX.
    """
    # Skip if Rivermax manager is not available
    manager_list = os.environ.get("ANO_MANAGER_LIST", "").split()
    if "rivermax" not in manager_list:
        pytest.skip("Rivermax manager not available in this build")

    # Skip if Rivermax license is not available
    # Rivermax license file is expected to be at /opt/mellanox/rivermax/rivermax.lic
    if not os.path.isfile("/opt/mellanox/rivermax/rivermax.lic"):
        pytest.skip("Rivermax license file not found")

    # Get the first two NICs for this test
    # Rivermax TX/RX test requires two NICs with IP addresses
    tx_interface = None
    rx_interface = None
    for i in range(len(nvidia_nics)):
        if nvidia_nics[i].ip_address:
            print(f"NVIDIA NIC {i} has IP address: {nvidia_nics[i].ip_address}")
            if tx_interface is None:
                tx_interface = nvidia_nics[i]
            else:
                rx_interface = nvidia_nics[i]
                break
        else:
            print(f"NVIDIA NIC {i} does not have IP address")
    if tx_interface is None or rx_interface is None:
        pytest.skip("NVIDIA NICs do not have IP addresses")

    # Generate media file
    media_file_path = os.path.join(work_dir, "test_media_file.ycbcr")
    generate_smpte2110_file(media_file_path)

    # Prepare config
    config_file = os.path.join(work_dir, "adv_networking_bench_rivermax_tx_rx.yaml")
    config_file_test = os.path.join(work_dir, "adv_networking_bench_rivermax_tx_rx_test.yaml")

    rx_settings_path = "advanced_network.cfg.interfaces[0].rx.queues[0].rivermax_rx_settings"
    tx_settings_path = "advanced_network.cfg.interfaces[0].tx.queues[0].rivermax_tx_settings"
    update_yaml_file(
        config_file,
        config_file_test,
        {
            "scheduler.max_duration_ms": 10000,
            f"{rx_settings_path}.settings_type": rivermax_receiver_type,
            f"{tx_settings_path}.local_ip_address": tx_interface.ip_address,
            f"{tx_settings_path}.destination_ip_address": "224.1.1.1",
            f"{tx_settings_path}.destination_port": 50001,
            "bench_tx.file_path": media_file_path,
        },
    )

    if rivermax_receiver_type == "ipo_receiver":
        update_yaml_file(
            config_file_test,
            config_file_test,
            {
                f"{rx_settings_path}.local_ip_addresses": [rx_interface.ip_address],
                f"{rx_settings_path}.source_ip_addresses": [tx_interface.ip_address],
                f"{rx_settings_path}.destination_ip_addresses": ["224.1.1.1"],
                f"{rx_settings_path}.destination_ports": [50001],
            },
        )
    else:
        update_yaml_file(
            config_file_test,
            config_file_test,
            {
                f"{rx_settings_path}.local_ip_address": rx_interface.ip_address,
                f"{rx_settings_path}.source_ip_address": tx_interface.ip_address,
                f"{rx_settings_path}.destination_ip_address": "224.1.1.1",
                f"{rx_settings_path}.destination_port": 50001,
            },
        )

    command = f"{executable} {config_file_test}"
    result = run_command(command, stream_output=True)
    results = parse_benchmark_results(result.stdout + result.stderr, "rivermax")

    received_packets = results.get_rx_packets(0)
    missed_pkts_check = results.get_missed_packets(0)
    errored_pkts_check = results.get_errored_packets(0)
    assert (
        received_packets > 0 and missed_pkts_check == 0 and errored_pkts_check == 0
    ), "Validation failed"
