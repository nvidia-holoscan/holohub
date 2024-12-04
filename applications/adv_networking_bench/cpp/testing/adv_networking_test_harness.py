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
from typing import List


def run_bash_cmd(cmd) -> CompletedProcess:
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


def validate_ano_benchmark(
    log: str,
    error_pkts_threshold: int,
    missed_pkts_threshold: int,
    avg_throughput_threshold: float,
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

    # Parse benchmark results
    transmit_pkts = int(re.search(r"Port 0.*?Transmit packets:\s+(\d+)", log, re.DOTALL).group(1))
    received_pkts = int(re.search(r"Port 1.*Received packets:\s+(\d+)", log, re.DOTALL).group(1))
    received_bytes = int(re.search(r"Port 1.*Received bytes:\s+(\d+)", log, re.DOTALL).group(1))
    missed_pkts = int(re.search(r"Port 1.*Missed packets:\s+(\d+)", log, re.DOTALL).group(1))
    errored_pkts = int(re.search(r"Port 1.*Errored packets:\s+(\d+)", log, re.DOTALL).group(1))
    exec_time = float(re.search(r"TOTAL EXECUTION TIME OF SCHEDULER : (\d+\.\d+) ms", log).group(1))

    logger.debug("Transmit packets:", transmit_pkts)
    logger.debug("Received packets:", received_pkts)
    logger.debug("Received bytes:", received_bytes)
    logger.debug("Missed packets:", missed_pkts)
    logger.debug("Errored packets:", errored_pkts)
    logger.debug("Exec time:", exec_time)

    # Check for missed packets
    # NOTE: default expects no missed packets, however current app isn't configured
    #   to consistently start tx or rx first, and there could be some dropped packets
    #   at startup based on the order of things. Using a slightly relaxed threshold
    #   based on the test duration is reasonable until that is addressed.
    missed_pkts_percent = (missed_pkts / transmit_pkts) * 100
    if missed_pkts_percent > missed_pkts_threshold:
        logger.error(
            f"Missed packets: {missed_pkts} ({missed_pkts_percent:.2f}%, over {missed_pkts_threshold}% threshold ❌)"
        )
        success = False
    else:
        logger.info(
            f"Missed packets: {missed_pkts} ({missed_pkts_percent:.2f}%, below {missed_pkts_threshold}% threshold ✅)"
        )

    # Check for errored packets
    errored_pkts_percent = (errored_pkts / transmit_pkts) * 100
    if errored_pkts_percent > error_pkts_threshold:
        logger.error(
            f"Errored packets: {errored_pkts} ({errored_pkts_percent:.2f}%, over {error_pkts_threshold}% threshold ❌)"
        )
        success = False
    else:
        logger.info(
            f"Errored packets: {errored_pkts} ({errored_pkts_percent:.2f}%, below {error_pkts_threshold}% threshold ✅)"
        )

    # Calculate average throughput
    # NOTE: This is an approximate and conservative heuristic, since:
    #   - exec_time includes the overhead from startup and shutdown of the holoscan graph
    #   - exec_time includes times when Rx might be up but Tx isn't yet sending
    #   An ideal test would trim the values at the beginning and end. A better tool
    #   is `mlnx_perf -i <interface>` but that would complicate this test wrapper.
    avg_throughput = (received_bytes * 8) / (exec_time / 1000) / 1e9
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
    args = parser.parse_args()

    # Run the bash command
    result = run_bash_cmd(args.command)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Validate the benchmark results
    success = validate_ano_benchmark(
        result.stdout + result.stderr,
        args.error_pkts_threshold,
        args.missed_pkts_threshold,
        args.avg_throughput_threshold,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
