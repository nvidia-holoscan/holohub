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

import json
import os
import signal
import subprocess
import sys
import time
import unittest

import requests


class TestTinyChat(unittest.TestCase):
    """Test cases for the TinyChat controller and model worker"""

    @classmethod
    def setUpClass(cls):
        """Start the TinyChat controller and model worker before running the tests"""
        # Start the controller
        cls.controller_process = subprocess.Popen(
            ["python3", "-m", "tinychat.serve.controller", "--host", "0.0.0.0", "--port", "10000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
            preexec_fn=os.setsid,
        )
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
        model_path = "/workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ/"
        quant_path = "/workspace/volumes/models/Llama-3-VILA1.5-8b-AWQ/llm/llama-3-vila1.5-8b-w4-g128-awq-v2.pt"

        cls.worker_process = subprocess.Popen(
            [
                "python3",
                "-m",
                "tinychat.serve.model_worker_new",
                "--host",
                "0.0.0.0",
                "--controller",
                "http://localhost:10000",
                "--port",
                "40000",
                "--worker",
                "http://localhost:40000",
                "--model-path",
                model_path,
                "--quant-path",
                quant_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        # Give the worker a moment to start and register with the controller
        time.sleep(10)

    @classmethod
    def tearDownClass(cls):
        """Stop the TinyChat controller and model worker after running the tests"""
        # Kill the processes
        if hasattr(cls, "worker_process"):
            os.killpg(os.getpgid(cls.worker_process.pid), signal.SIGTERM)
        if hasattr(cls, "controller_process"):
            os.killpg(os.getpgid(cls.controller_process.pid), signal.SIGTERM)

        # Give them a moment to shut down
        time.sleep(10)


if __name__ == "__main__":
    unittest.main()
