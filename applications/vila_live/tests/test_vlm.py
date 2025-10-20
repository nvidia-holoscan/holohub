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
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add the parent directory to the path so we can import the VLM class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vila_live import V4L2toVLM

expected_image_size = 150000  # frame buffer size
expected_prompt_size = 322


class MockVLMHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler that simulates the VLM server responses"""

    def do_POST(self):
        """Handle POST requests to /worker_generate_stream"""
        if self.path != "/worker_generate_stream":
            assert False, "unknown request"
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        request = json.loads(post_data.decode("utf-8"))
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        # Get the prompt from the request
        prompt = request.get("prompt", "")
        # Create a mock response
        response_text = prompt + "This is a mock response from the VLM server."
        assert (
            len(request["prompt"]) == expected_prompt_size
        ), f"Prompt length is {len(request['prompt'])}"
        assert (
            len(request["images"][0]) >= expected_image_size
        ), f"Image length is {len(request['images'][0])}"
        # Send the response in chunks to simulate streaming
        for i in range(3):
            chunk = {"error_code": 0, "text": response_text[: len(prompt) + (i + 1) * 10]}
            self.wfile.write(json.dumps(chunk).encode() + b"\0")


class TestVLM(unittest.TestCase):
    """Test cases for the VLM class"""

    @classmethod
    def setUpClass(cls):
        """Start a mock HTTP server before running the tests"""
        cls.server = HTTPServer(("localhost", 40000), MockVLMHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        # Give the server a moment to start
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        """Stop the mock HTTP server after running the tests"""
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def test_generate_response(self):
        """Test the generate_response method"""
        app = V4L2toVLM("none", "replayer", "none")
        testing_yaml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vila_live_testing.yaml"
        )
        if not os.path.exists(testing_yaml):
            raise FileNotFoundError(f"Testing YAML file not found: {testing_yaml}")
        app.config(testing_yaml)
        app.run()


if __name__ == "__main__":
    unittest.main()
