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
import argparse
import logging
import re
import shlex
import subprocess
import sys
from subprocess import CalledProcessError, CompletedProcess
from threading import Thread
from time import sleep
from typing import Dict, List, Tuple


def run_bash_cmd(cmd, external_script: str) -> CompletedProcess:
    """
    Runs a bash command and captures its output.

    Args:
        cmd (str or list): The command to run.

    Returns:
        CompletedProcess: The result of the command execution.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Running command: {cmd}")

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    # Convert the bash command to a list of arguments if needed,
    # so as not to require Shell=True
    if isinstance(cmd, str):
        args = shlex.split(cmd)
    else:
        args = cmd

    # Start process
    try:
        p = subprocess.Popen(
            args,
            cwd=None,
            universal_newlines=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        logger.error(f"Failed to create process: {e}")
        raise

    if external_script is not None:
        # Define a function to run the external script after a delay
        def delayed_external_script():
            logger.info("Sleeping before launching script")
            sleep(5)
            try:
                external_args = shlex.split(external_script)
                subprocess.run(
                    external_args,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                logger.error(f"Failed to run external script: {e}")
                raise

        # Start the thread for the delayed execution of the external script
        Thread(target=delayed_external_script).start()
        logger.info("Launched external script")

    # Stream & capture log
    while p.poll() is None:
        for line in iter(p.stderr.readline, ""):
            print(line.strip(), flush=True)
            stderr_lines.append(line)
        for line in iter(p.stdout.readline, ""):
            print(line.strip(), flush=True)
            stdout_lines.append(line)

    # Capture any remaining output after process ends
    for line in p.stderr:
        print(line.strip(), flush=True)
        stderr_lines.append(line)
    for line in p.stdout:
        print(line.strip(), flush=True)
        stdout_lines.append(line)

    # Close pipes
    p.stdout.close()
    p.stderr.close()

    # Return
    std_out = "".join(stdout_lines)
    std_err = "".join(stderr_lines)
    rc = p.poll()
    if rc != 0:
        logger.debug(f"Command failed with exit code {rc}")
        raise CalledProcessError(rc, cmd, std_out, std_err)
    return CompletedProcess(cmd, rc, std_out, std_err)


def compare_metrics(name: str, input: List[Tuple[str, str]], expected: List[int]) -> bool:
    if expected is None:
        return True

    logger = logging.getLogger(__name__)

    if len(input) != len(expected):
        logger.error(f"{name} metrics size doesn't match actual: {len(input)/len(expected)}")
        return False

    for idx, el in enumerate(input):
        if int(el[1]) != expected[idx]:
            logger.error(
                f"{name} index {idx} failed match: "
                f"expected={expected[idx]}, actual={int(el[1])}"
            )
            return False
        else:
            logger.debug(
                f"{name} index {idx} passed match: "
                f"expected={expected[idx]}, actual={int(el[1])}"
            )

    return True


def parse_port_map(port_map_str: str) -> Dict[int, int]:
    pm = {}
    pairs = port_map_str.split(",")
    for p in pairs:
        pair = p.split("-")
        pm[int(pair[0])] = int(pair[1])

    return pm


def validate_ano_benchmark(
    log: str,
    port_map: str,
    error_pkts_threshold: int,
    missed_pkts_threshold: int,
    avg_throughput_threshold: float,
    expected_q_pkts: List[int] = None,
) -> bool:
    """
    Validates the benchmark results.

    Args:
        log (str): The output of the benchmark run.
        error_pkts_threshold (int): Maximum allowed errored packets.
        missed_pkts_threshold (int): Maximum allowed missed packets.
        avg_throughput_threshold (float): Minimum required average throughput in Gbps.
    Returns:
        bool: True if the benchmark passed, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Validating benchmark results")

    success = True
    # Check for errors in the log
    if "[error]" in log:
        logger.error("Errors found in benchmark output")
        success = False

    pm = parse_port_map(port_map)
    logger.info(f"Port map is: {pm}")

    # Parse benchmark results
    transmit_pkts = re.findall(r"Port (\d+).*?Transmit packets:\s+(\d+)", log, re.DOTALL)
    received_pkts = re.findall(r"Port (\d+).*?Received packets:\s+(\d+)", log, re.DOTALL)
    received_bytes = re.findall(r"Port (\d+).*?Received bytes:\s+(\d+)", log, re.DOTALL)
    missed_pkts = re.findall(r"Port (\d+).*?Missed packets:\s+(\d+)", log, re.DOTALL)
    errored_pkts = re.findall(r"Port (\d+).*?Errored packets:\s+(\d+)", log, re.DOTALL)
    q_rx_pkts = re.findall(r"rx_q(\d+)_packets:\s+(\d+)", log, re.DOTALL)

    if not compare_metrics("q_rx_pkts", q_rx_pkts, expected_q_pkts):
        logger.info("Failed q_rx_pkts check")

    exec_time = float(re.search(r"TOTAL EXECUTION TIME OF SCHEDULER : (\d+\.\d+) ms", log).group(1))
    logger.debug("Transmit packets:", transmit_pkts)
    logger.debug("Received packets:", received_pkts)
    logger.debug("Received bytes:", received_bytes)
    logger.debug("Missed packets:", missed_pkts)
    logger.debug("Errored packets:", errored_pkts)
    logger.debug("Exec time:", exec_time)

    for tx, rx in pm.items():
        logger.info(f"Processing port pair: {tx}->{rx}")

        # For now we assume the port is in increasing order starting from 0
        missed_pkts_port = int(missed_pkts[rx][1])
        transmit_pkts_port = int(transmit_pkts[tx][1])
        error_pkts_port = int(errored_pkts[rx][1])
        received_bytes_port = int(received_bytes[rx][1])

        # Check for missed packets
        # NOTE: default expects no missed packets, however current app isn't configured
        #   to consistently start tx or rx first, and there could be some dropped packets
        #   at startup based on the order of things. Using a slightly relaxed threshold
        #   based on the test duration is reasonable until that is addressed.
        if transmit_pkts_port > 0:
            missed_pkts_percent = (missed_pkts_port / transmit_pkts_port) * 100
            if missed_pkts_percent > missed_pkts_threshold:
                logger.error(
                    f"Missed packets: {missed_pkts_port} ({missed_pkts_percent:.2f}%, over {missed_pkts_threshold}% threshold ❌)"
                )
                success = False
            else:
                logger.info(
                    f"Missed packets: {missed_pkts_port} ({missed_pkts_percent:.2f}%, below {missed_pkts_threshold}% threshold ✅)"
                )

            # Check for errored packets
            errored_pkts_percent = (error_pkts_port / transmit_pkts_port) * 100
            if errored_pkts_percent > error_pkts_threshold:
                logger.error(
                    f"Errored packets: {error_pkts_port} ({errored_pkts_percent:.2f}%, over {error_pkts_threshold}% threshold ❌)"
                )
                success = False
            else:
                logger.info(
                    f"Errored packets: {error_pkts_port} ({errored_pkts_percent:.2f}%, below {error_pkts_threshold}% threshold ✅)"
                )

        # Calculate average throughput
        # NOTE: This is an approximate and conservative heuristic, since:
        #   - exec_time includes the overhead from startup and shutdown of the holoscan graph
        #   - exec_time includes times when Rx might be up but Tx isn't yet sending
        #   An ideal test would trim the values at the beginning and end. A better tool
        #   is `mlnx_perf -i <interface>` but that would complicate this test wrapper.
        avg_throughput = (received_bytes_port * 8) / (exec_time / 1000) / 1e9
        if avg_throughput < avg_throughput_threshold:
            logger.error(
                f"Average throughput: {avg_throughput:.2f} Gbps (below {avg_throughput_threshold} Gbps threshold ❌)"
            )
            success = False
        else:
            logger.info(
                f"Average throughput: {avg_throughput:.2f} Gbps (above {avg_throughput_threshold} Gbps threshold ✅)"
            )

    return success


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a bash command")
    parser.add_argument("command", help="The bash command to run")
    parser.add_argument(
        "--error-packets-threshold",
        dest="error_pkts_threshold",
        type=float,
        default=0.0,
        help="Maximum allowed errored packets percentage",
    )
    parser.add_argument(
        "--missed-packets-threshold",
        dest="missed_pkts_threshold",
        type=float,
        default=0.0,
        help="Maximum allowed missed packets percentage",
    )
    parser.add_argument(
        "--avg-throughput-threshold",
        type=float,
        default=95,
        help="Minimum required average throughput in Gbps",
    )
    parser.add_argument(
        "--packets-per-rx-queue",
        type=int,
        dest="packets_per_rx_queue",
        nargs="+",
        help="Expected number of packets per RX queue",
    )
    parser.add_argument(
        "--external-script",
        type=str,
        dest="external_script",
        default=None,
        help="External script to execute after ANO has started",
    )
    parser.add_argument(
        "--port-map",
        type=str,
        dest="port_map",
        default="",
        help="Port mapping for TX/RX pairs. Format of 0-1,1-2 For 0->1 and 1->2",
    )
    args = parser.parse_args()

    # Run the bash command
    try:
        result = run_bash_cmd(args.command, args.external_script)
        if result.returncode != 0:
            sys.exit(result.returncode)
    except Exception as e:
        logger.error(f"Exiting due to external script error: {e}")
        sys.exit(-1)

    # Validate the benchmark results
    success = validate_ano_benchmark(
        result.stdout + result.stderr,
        args.port_map,
        args.error_pkts_threshold,
        args.missed_pkts_threshold,
        args.avg_throughput_threshold,
        args.packets_per_rx_queue,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
