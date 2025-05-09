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
import re
import sys
from typing import Dict

# Configure the logger
logger = logging.getLogger(__name__)


class BenchmarkResults:
    """Class to hold parsed benchmark results and provide analysis methods."""

    def __init__(
        self,
        tx_pkts: Dict[str, int],
        tx_bytes: Dict[str, int],
        rx_pkts: Dict[str, int],
        rx_bytes: Dict[str, int],
        missed_pkts: Dict[str, int],
        errored_pkts: Dict[str, int],
        q_rx_pkts: Dict[str, Dict[str, int]],
        q_tx_pkts: Dict[str, Dict[str, int]],
        exec_time: float,
    ):
        """
        Initialize benchmark results from parsed values.

        Args:
            tx_pkts: Dictionary mapping port number (string) to transmitted packet count
            tx_bytes: Dictionary mapping port number (string) to transmitted byte count
            rx_pkts: Dictionary mapping port number (string) to received packet count
            rx_bytes: Dictionary mapping port number (string) to received byte count
            missed_pkts: Dictionary mapping port number (string) to missed packet count
            errored_pkts: Dictionary mapping port number (string) to errored packet count
            q_rx_pkts: Nested dictionary mapping port to queue to packet count {port: {queue: count}}
            q_tx_pkts: Nested dictionary mapping port to queue to packet count {port: {queue: count}}
            exec_time: Execution time in milliseconds
        """
        self.tx_pkts = tx_pkts
        self.tx_bytes = tx_bytes
        self.rx_pkts = rx_pkts
        self.rx_bytes = rx_bytes
        self.missed_pkts = missed_pkts
        self.errored_pkts = errored_pkts
        self.q_rx_pkts = q_rx_pkts
        self.q_tx_pkts = q_tx_pkts
        self.exec_time = exec_time

    def get_tx_packets(self, port: int) -> int:
        """Get number of transmitted packets for a port."""
        return self.tx_pkts.get(str(port), 0)

    def get_tx_bytes(self, port: int) -> int:
        """Get number of transmitted bytes for a port."""
        return self.tx_bytes.get(str(port), 0)

    def get_rx_packets(self, port: int) -> int:
        """Get number of received packets for a port."""
        return self.rx_pkts.get(str(port), 0)

    def get_rx_bytes(self, port: int) -> int:
        """Get number of received bytes for a port."""
        return self.rx_bytes.get(str(port), 0)

    def get_missed_packets(self, port: int) -> int:
        """Get number of missed packets for a port."""
        return self.missed_pkts.get(str(port), 0)

    def get_errored_packets(self, port: int) -> int:
        """Get number of errored packets for a port."""
        return self.errored_pkts.get(str(port), 0)

    def get_rx_queue_packets(self, port: int, queue: int) -> int:
        """
        Get number of received packets for a specific queue on a port.

        Args:
            port: Port number
            queue: Queue number

        Returns:
            int: Number of packets received on the queue
        """
        port_str = str(port)
        queue_str = str(queue)
        if port_str not in self.q_rx_pkts:
            return 0
        return self.q_rx_pkts[port_str].get(queue_str, 0)

    def get_tx_queue_packets(self, port: int, queue: int) -> int:
        """
        Get number of transmitted packets for a specific queue on a port.

        Args:
            port: Port number
            queue: Queue number

        Returns:
            int: Number of packets transmitted on the queue
        """
        port_str = str(port)
        queue_str = str(queue)
        if port_str not in self.q_tx_pkts:
            return 0
        return self.q_tx_pkts[port_str].get(queue_str, 0)

    def get_rx_throughput(self, port: int) -> float:
        """
        Calculate average receive throughput for a port in Gbps.

        Args:
            port: Port number to calculate throughput for

        Returns:
            float: Average throughput in Gbps (Gigabits per second)
        """
        rx_bytes = self.get_rx_bytes(port)
        if rx_bytes == 0 or self.exec_time == 0:
            return 0.0

        # Convert bytes to bits, and divide by time (converted from ms to seconds)
        # to get bps, then convert to Gbps
        return (rx_bytes * 8) / (self.exec_time / 1000) / 1e9

    def get_tx_throughput(self, port: int) -> float:
        """
        Calculate average transmit throughput for a port in Gbps.

        Args:
            port: Port number to calculate throughput for

        Returns:
            float: Average throughput in Gbps (Gigabits per second)
        """
        tx_bytes = self.get_tx_bytes(port)
        if tx_bytes == 0 or self.exec_time == 0:
            return 0.0

        # Convert bytes to bits, and divide by time (converted from ms to seconds)
        # to get bps, then convert to Gbps
        return (tx_bytes * 8) / (self.exec_time / 1000) / 1e9

    def validate_rx_queue_packets(
        self, port: int, expected_packets: Dict[int, int], gt: bool = False
    ) -> bool:
        """
        Validate that RX queues on a specific port received the expected number of packets.

        Args:
            port: Port number to validate
            expected_packets: Dictionary mapping queue numbers to expected packet counts {queue: count}
            gt: If True, validate that the actual packet count is greater than the expected count

        Returns:
            bool: True if validation passed, False otherwise
        """
        port_str = str(port)
        success = True
        if port_str not in self.q_rx_pkts:
            if expected_packets:  # If we expected packets but found none
                logger.error(f"No RX queue packets found for port {port}")
                return False
            return True  # If we didn't expect any packets and found none

        # Convert expected_packets keys to strings for comparison
        expected_str_dict = {str(q): count for q, count in expected_packets.items()}

        # Check each expected queue
        for queue_str, expected_count in expected_str_dict.items():
            actual_count = self.q_rx_pkts[port_str].get(queue_str, 0)

            if gt:
                if actual_count <= expected_count:
                    logger.error(
                        f"Port {port} Queue {queue_str} packet count mismatch: "
                        f"expected (at least) ={expected_count}, actual={actual_count} ❌"
                    )
                    success = False
                else:
                    logger.info(
                        f"Port {port} Queue {queue_str} packet count match: "
                        f"expected (at least) {expected_count}, actual={actual_count} ✅"
                    )
            else:
                if actual_count != expected_count:
                    logger.error(
                        f"Port {port} Queue {queue_str} packet count mismatch: "
                        f"expected={expected_count}, actual={actual_count} ❌"
                    )
                    success = False
                else:
                    logger.info(
                        f"Port {port} Queue {queue_str} packet count match: "
                        f"expected={expected_count}, actual={actual_count} ✅"
                    )

        # Check for unexpected queues with packets
        for queue_str, actual_count in self.q_rx_pkts[port_str].items():
            if queue_str not in expected_str_dict and actual_count > 0:
                logger.warning(
                    f"Port {port} Queue {queue_str} has unexpected packets: {actual_count} ⚠️"
                )

        return success

    def validate_missed_packets(self, port_map: Dict[int, int], threshold: float) -> bool:
        """
        Validate missed packets are below threshold.

        Args:
            port_map: Dictionary mapping TX ports to RX ports (e.g., {0: 1})
            threshold: Maximum allowed percentage of missed packets

        Returns:
            bool: True if validation passed, False otherwise
        """
        success = True

        for tx_port, rx_port in port_map.items():
            missed_rx_pkts = self.get_missed_packets(rx_port)
            tx_pkts = self.get_tx_packets(tx_port)

            if tx_pkts > 0:
                missed_pkts_percent = (missed_rx_pkts / tx_pkts) * 100
                if missed_pkts_percent > threshold:
                    logger.error(
                        f"Missed packets: {missed_rx_pkts} ({missed_pkts_percent:.2f}%, over {threshold}% threshold ❌)"
                    )
                    success = False
                else:
                    logger.info(
                        f"Missed packets: {missed_rx_pkts} ({missed_pkts_percent:.2f}%, below {threshold}% threshold ✅)"
                    )

        return success

    def validate_errored_packets(self, port_map: Dict[int, int], threshold: float) -> bool:
        """
        Validate errored packets are below threshold.

        Args:
            port_map: Dictionary mapping TX ports to RX ports (e.g., {0: 1})
            threshold: Maximum allowed percentage of errored packets

        Returns:
            bool: True if validation passed, False otherwise
        """
        success = True

        for tx_port, rx_port in port_map.items():
            errored_rx_pkts = self.get_errored_packets(rx_port)
            tx_pkts = self.get_tx_packets(tx_port)

            if tx_pkts > 0:
                errored_pkts_percent = (errored_rx_pkts / tx_pkts) * 100
                if errored_pkts_percent > threshold:
                    logger.error(
                        f"Errored packets: {errored_rx_pkts} ({errored_pkts_percent:.2f}%, over {threshold}% threshold ❌)"
                    )
                    success = False
                else:
                    logger.info(
                        f"Errored packets: {errored_rx_pkts} ({errored_pkts_percent:.2f}%, below {threshold}% threshold ✅)"
                    )

        return success

    def validate_throughput(self, port_map: Dict[int, int], threshold: float) -> bool:
        """
        Validate throughput is above threshold.

        Args:
            port_map: Dictionary mapping TX ports to RX ports (e.g., {0: 1})
            threshold: Minimum required throughput in Gbps

        Returns:
            bool: True if validation passed, False otherwise
        """
        success = True

        for tx_port, rx_port in port_map.items():
            rx_throughput = self.get_rx_throughput(rx_port)

            if rx_throughput < threshold:
                logger.error(
                    f"Average throughput: {rx_throughput:.2f} Gbps (below {threshold} Gbps threshold ❌)"
                )
                success = False
            else:
                logger.info(
                    f"Average throughput: {rx_throughput:.2f} Gbps (above {threshold} Gbps threshold ✅)"
                )

        return success


# Helper function to extract metrics
def _extract_metric(pattern, text) -> int:
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return 0  # Return 0 if no match found


def parse_benchmark_results(log: str, manager_type: str) -> BenchmarkResults:
    if manager_type == "dpdk":
        return parse_dpdk_benchmark_results(log)
    elif manager_type == "gpunetio":
        return parse_gpunetio_benchmark_results(log)
    else:
        raise ValueError(f"Unsupported manager type: {manager_type}")


def parse_dpdk_benchmark_results(log: str) -> BenchmarkResults:
    """
    Parse benchmark results from DPDK log output.

    Args:
        log: The log output as a string

    Returns:
        BenchmarkResults: A structured representation of the benchmark results
    """
    # Initialize result dictionaries
    tx_packets = {}
    tx_bytes = {}
    rx_packets = {}
    rx_bytes = {}
    missed_packets = {}
    errored_packets = {}
    rx_queue_packets = {}
    tx_queue_packets = {}

    # Regex patterns
    port_pattern = r"Port (\d+):"
    metric_patterns = {
        "rx_packets": (r"Received packets:\s+(\d+)", rx_packets),
        "tx_packets": (r"Transmit packets:\s+(\d+)", tx_packets),
        "rx_bytes": (r"Received bytes:\s+(\d+)", rx_bytes),
        "tx_bytes": (r"Transmit bytes:\s+(\d+)", tx_bytes),
        "missed_packets": (r"Missed packets:\s+(\d+)", missed_packets),
        "errored_packets": (r"Errored packets:\s+(\d+)", errored_packets),
    }
    rx_queue_pattern = r"rx_q(\d+)_packets:\s+(\d+)"
    tx_queue_pattern = r"tx_q(\d+)_packets:\s+(\d+)"
    exec_time_pattern = r"TOTAL EXECUTION TIME OF SCHEDULER : (\d+\.\d+) ms"

    # Find all port sections in the log
    port_matches = list(re.finditer(port_pattern, log))
    logger.debug(f"Number of port_matches: {len(port_matches)}")

    for i, port_match in enumerate(port_matches):
        # Extract the port ID from the match
        port_id = port_match.group(1)
        logger.debug(f"Parsing stats for port ID: {port_id}")

        # Extract the stats for this port
        port_start = port_match.end()
        if i < len(port_matches) - 1:
            port_end = port_matches[i + 1].start()  # Stop at next port's start
        else:
            port_end = len(log)  # last port: go to end of log
        port_section = log[port_start:port_end]

        # Extract all metrics for this port
        for metric_name, (pattern, result_dict) in metric_patterns.items():
            value = _extract_metric(pattern, port_section)
            result_dict[port_id] = value
            logger.debug(f"Port {port_id} - {metric_name}: {value}")

        # Parse queue-specific information
        rx_queue_matches = re.findall(rx_queue_pattern, port_section)
        logger.debug(f"Port {port_id} - rx_queue_matches: {rx_queue_matches}")

        for queue_id, count in rx_queue_matches:
            if port_id not in rx_queue_packets:
                rx_queue_packets[port_id] = {}
            rx_queue_packets[port_id][queue_id] = int(count)

        tx_queue_matches = re.findall(tx_queue_pattern, port_section)
        logger.debug(f"Port {port_id} - tx_queue_matches: {tx_queue_matches}")

        for queue_id, count in tx_queue_matches:
            if port_id not in tx_queue_packets:
                tx_queue_packets[port_id] = {}
            tx_queue_packets[port_id][queue_id] = int(count)

    # Extract execution time
    exec_time = 0.0
    exec_time_match = re.search(exec_time_pattern, log)
    if exec_time_match:
        exec_time = float(exec_time_match.group(1))

    # Debug output
    logger.debug(f"TX packets: {tx_packets}")
    logger.debug(f"TX bytes: {tx_bytes}")
    logger.debug(f"RX packets: {rx_packets}")
    logger.debug(f"RX bytes: {rx_bytes}")
    logger.debug(f"Missed packets: {missed_packets}")
    logger.debug(f"Errored packets: {errored_packets}")
    logger.debug(f"RX queue packets: {rx_queue_packets}")
    logger.debug(f"TX queue packets: {tx_queue_packets}")
    logger.debug(f"Exec time: {exec_time}")

    return BenchmarkResults(
        tx_pkts=tx_packets,
        tx_bytes=tx_bytes,
        rx_pkts=rx_packets,
        rx_bytes=rx_bytes,
        missed_pkts=missed_packets,
        errored_pkts=errored_packets,
        q_rx_pkts=rx_queue_packets,
        q_tx_pkts=tx_queue_packets,
        exec_time=exec_time,
    )


def parse_gpunetio_benchmark_results(log: str) -> BenchmarkResults:
    """
    Parse benchmark results from GPUNetIO log output.

    Args:
        log: The log output as a string

    Returns:
        BenchmarkResults: A structured representation of the benchmark results
    """
    # Initialize result dictionaries - GPUNetIO logs seem to provide totals only.
    # We'll store these totals under a default port "0".
    tx_packets = {}
    tx_bytes = {}
    rx_packets = {}
    rx_bytes = {}
    missed_packets = {}
    errored_packets = {}  # Not provided in logs
    rx_queue_packets = {}  # Not provided in logs
    tx_queue_packets = {}  # Not provided in logs

    # Regex patterns for GPUNetIO
    rx_pkts_pattern = r"Total Rx packets\s+(\d+)"
    rx_bytes_pattern = r"Total Rx bytes\s+(\d+)"
    tx_pkts_pattern = r"Total Tx packets\s+(\d+)"
    tx_bytes_pattern = r"Total Tx bytes\s+(\d+)"
    # Assuming exec time format is the same as DPDK
    exec_time_pattern = r"TOTAL EXECUTION TIME OF SCHEDULER : (\d+\.\d+) ms"

    # Extract total metrics and assign to port "0"
    rx_packets["0"] = _extract_metric(rx_pkts_pattern, log)
    rx_bytes["0"] = _extract_metric(rx_bytes_pattern, log)
    tx_packets["0"] = _extract_metric(tx_pkts_pattern, log)
    tx_bytes["0"] = _extract_metric(tx_bytes_pattern, log)

    # Calculate missed packets for GPUNetIO
    missed_packets["0"] = max(0, tx_packets.get("0", 0) - rx_packets.get("0", 0))

    # Extract execution time
    exec_time = 0.0
    exec_time_match = re.search(exec_time_pattern, log)
    if exec_time_match:
        exec_time = float(exec_time_match.group(1))

    # Debug output
    logger.debug(f"Total RX packets (assigned to port 0): {rx_packets.get('0', 0)}")
    logger.debug(f"Total RX bytes (assigned to port 0): {rx_bytes.get('0', 0)}")
    logger.debug(f"Total TX packets (assigned to port 0): {tx_packets.get('0', 0)}")
    logger.debug(f"Total TX bytes (assigned to port 0): {tx_bytes.get('0', 0)}")
    logger.debug(f"Missed packets: {missed_packets}")
    logger.debug(f"Errored packets: {errored_packets}")
    logger.debug(f"RX queue packets: {rx_queue_packets}")
    logger.debug(f"TX queue packets: {tx_queue_packets}")
    logger.debug(f"Exec time: {exec_time}")

    return BenchmarkResults(
        tx_pkts=tx_packets,
        tx_bytes=tx_bytes,
        rx_pkts=rx_packets,
        rx_bytes=rx_bytes,
        missed_pkts=missed_packets,
        errored_pkts=errored_packets,
        q_rx_pkts=rx_queue_packets,
        q_tx_pkts=tx_queue_packets,
        exec_time=exec_time,
    )


if __name__ == "__main__":
    # Set up console logging if run directly
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("This module is designed to be imported, not run directly.")
    sys.exit(0)
