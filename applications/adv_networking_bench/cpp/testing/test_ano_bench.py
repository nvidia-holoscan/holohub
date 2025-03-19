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

import os
import pytest
import sys
import logging
from threading import Thread
from time import sleep
import os.path

# Import the test harness functionality
from test_ano_bench_utils import (
    start_bash_cmd,
    monitor_process,
    parse_benchmark_results
)

# Configure the logger
logger = logging.getLogger(__name__)


def test_multi_if_loopback(work_dir):
    """
    Test 1: TX/RX loopback over single link with one TX queue and one RX queue.

    This test focuses on:
    - Missed packets staying below threshold
    - Errored packets staying below threshold
    - Average throughput staying above threshold
    """
    # Parameters for this test
    config_file = "adv_networking_bench_dpdk_multi_if_loopback_test.yaml"
    port_map = {0: 1}  # Port 0 (TX) sends to Port 1 (RX) # TODO: infer from config file?
    avg_throughput_threshold = 90.0
    missed_pkts_threshold = 0.1
    error_pkts_threshold = 0.0

    # Build the command
    executable = os.path.join(work_dir, "adv_networking_bench")
    command = f"{executable} {config_file}"

    # Run the application until completion and parse the results
    p = start_bash_cmd(command)
    result = monitor_process(p, command)
    results = parse_benchmark_results(result.stdout + result.stderr)

    # Validate the following metrics
    assert results.validate_missed_packets(port_map, missed_pkts_threshold), "Missed packets validation failed"
    assert results.validate_errored_packets(port_map, error_pkts_threshold), "Errored packets validation failed"
    assert results.validate_throughput(port_map, avg_throughput_threshold), "Throughput validation failed"


def test_multi_rx_q(work_dir):
    """
    Test 2: RX multi queue with a single CPU core using scapy to send packets.

    This test focuses on:
    - Validating the distribution of packets to RX queues
    """
    # Parameters for this test
    config_file = "adv_networking_bench_dpdk_rx_multi_q.yaml"
    port_map = {0: 0}  # Port 0 (TX) sends to Port 0 (RX) - loopback
    expected_q_pkts = {0: 1, 1: 1}  # Expecting 1 packet for queue 0 and 1

    # Build the absolute path to the executable
    executable = os.path.join(work_dir, "adv_networking_bench")
    command = f"{executable} {config_file}"

    # Run the application (non-blocking)
    p = start_bash_cmd(command)

    # Send packets with scapy after a 5s delay
    def send_test_packets():
        logger.info("Sleeping before sending test packets")
        sleep(5)
        try:
            from scapy.all import IP, UDP, Ether, sendp

            logger.info("Sending test packets to queues")
            packet1 = Ether() / IP(dst="10.10.100.2") / UDP(sport=4095, dport=4095) / ("X" * (1050 - 20 - 8))
            packet2 = Ether() / IP(dst="10.10.100.2") / UDP(sport=4096, dport=4096) / ("X" * (1050 - 20 - 8))

            # Send one packet to each queue
            sendp(packet1, iface="cx7_0", count=1, verbose=1)
            sendp(packet2, iface="cx7_0", count=1, verbose=1)
            logger.info("Test packets sent successfully")
        except Exception as e:
            logger.error(f"Failed to send test packets: {e}")
            raise

    # Start the packet sending thread (non-blocking)
    Thread(target=send_test_packets).start()

    # Monitor the application until completion and parse the results
    result = monitor_process(p, command)
    results = parse_benchmark_results(result.stdout + result.stderr)

    # For this test, we only care about queue packet distribution (on port 0)
    assert results.validate_rx_queue_packets(0, expected_q_pkts)