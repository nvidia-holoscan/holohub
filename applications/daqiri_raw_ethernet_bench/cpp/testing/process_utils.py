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
import shlex
import subprocess
from subprocess import CalledProcessError, CompletedProcess

# Configure the logger
logger = logging.getLogger(__name__)


def run_command(cmd: str, stream_output: bool = False) -> CompletedProcess:
    """
    Run a shell command synchronously and return the result.
    This function captures the output but doesn't display it in real-time.

    Args:
        cmd: Command to run
        stream_output: Whether to stream the output to the console
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if stream_output:
        p = start_process(cmd)
        return monitor_process(p)
    else:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
            check=False,
        )


def start_process(cmd) -> subprocess.Popen:
    """
    Start a process and return the process object without waiting for completion.
    The command is stored in the process object for later reference.

    Args:
        cmd (str or list): The command to run.

    Returns:
        subprocess.Popen: The process object for the running command.
    """
    logger.debug(f"Starting process: {cmd}")

    # Convert the command to a list of arguments if needed
    if isinstance(cmd, str):
        args = shlex.split(cmd)
    else:
        args = cmd

    # Store the original command for reference
    if isinstance(cmd, str):
        orig_cmd = cmd
    else:
        orig_cmd = " ".join(cmd)

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
        # Store the command in the process object
        p._orig_cmd = orig_cmd
        return p
    except Exception as e:
        logger.error(f"Failed to create process: {e}")
        raise


def monitor_process(p) -> CompletedProcess:
    """
    Monitor a process until completion, capturing its stdout and stderr.
    This function streams the output to the console in real-time while also capturing it.

    Args:
        p (subprocess.Popen): The process to monitor.

    Returns:
        CompletedProcess: The result of the command execution.

    Raises:
        CalledProcessError: If the process returns a non-zero exit code.
    """
    stdout_lines = []
    stderr_lines = []

    # Stream & capture log through pipes
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

    # Get the command that was used to start the process
    cmd = getattr(p, "_orig_cmd", "unknown command")

    # Return
    std_out = "".join(stdout_lines)
    std_err = "".join(stderr_lines)

    rc = p.poll()
    if rc != 0:
        logger.debug(f"Command failed with exit code {rc}")
        raise CalledProcessError(rc, cmd, std_out, std_err)
    return CompletedProcess(cmd, rc, std_out, std_err)


if __name__ == "__main__":
    # Set up console logging if run directly
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("This module is designed to be imported, not run directly.")
