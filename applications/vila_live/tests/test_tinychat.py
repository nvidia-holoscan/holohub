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

import os
import signal
import subprocess
import time
import unittest

import requests
import yaml

# Use environment variable if set, otherwise fallback to source directory location
test_yaml = os.environ.get(
    "VILA_LIVE_TEST_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vila_live_testing.yaml"),
)


class TestTinyChat(unittest.TestCase):
    """Test cases for the TinyChat controller and model worker"""

    @classmethod
    def setUpClass(cls):
        """Start the TinyChat controller and model worker before running the tests"""
        # Start the controller
        print("Starting controller process...")
        cls.controller_process = subprocess.Popen(
            ["python3", "-m", "tinychat.serve.controller", "--host", "0.0.0.0", "--port", "10000"],
            stdout=None,  # Use None to inherit the parent's stdout
            stderr=None,  # Use None to inherit the parent's stderr
            env=os.environ,
            preexec_fn=os.setsid,
        )
        print(f"Controller process started with PID: {cls.controller_process.pid}")
        time.sleep(10)
        try:
            response = requests.get("http://localhost:10000")
            if response.status_code != 404:
                cls.tearDownClass()
                raise Exception("Controller failed to start properly")
        except requests.exceptions.ConnectionError:
            cls.tearDownClass()
            raise Exception("Could not connect to controller")

        # Start the model worker
        print("Starting model worker process...")
        with open(test_yaml, "r") as f:
            config = yaml.safe_load(f)
        model_path = config["model_path"]
        quant_path = config["quant_path"]

        cmd = (
            "python3 -m tinychat.serve.model_worker_new "
            "--host 0.0.0.0 --controller http://localhost:10000 --port 40000 "
            "--worker http://localhost:40000 "
            f"--model-path {model_path} --quant-path {quant_path}"
        )
        cls.worker_process = subprocess.Popen(
            cmd.split(),
            stdout=None,  # Use None to inherit the parent's stdout
            stderr=None,  # Use None to inherit the parent's stderr
            preexec_fn=os.setsid,
        )
        print(f"Worker process started with PID: {cls.worker_process.pid}")
        # Give the worker a moment to start and register with the controller
        time.sleep(10)

    def test_tinychat(self):
        """Test the TinyChat functionality"""
        print(
            f"Testing processes started...{self.controller_process.pid} {self.worker_process.pid}"
        )

    @classmethod
    def tearDownClass(cls):
        """Stop the TinyChat controller and model worker after running the tests"""
        # Kill processes directly
        if hasattr(cls, "worker_process"):
            try:
                os.kill(cls.worker_process.pid, signal.SIGKILL)
                print("Worker process killed")
            except Exception as e:
                print(f"Error killing worker process: {e}")

        if hasattr(cls, "controller_process"):
            try:
                os.kill(cls.controller_process.pid, signal.SIGKILL)
                print("Controller process killed")
            except Exception as e:
                print(f"Error killing controller process: {e}")


if __name__ == "__main__":
    unittest.main()
