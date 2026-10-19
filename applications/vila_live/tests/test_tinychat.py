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

import base64
import json
import os
import shutil
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
            start_new_session=True,
        )
        cls.addClassCleanup(cls._stop_processes)
        print(f"Controller process started with PID: {cls.controller_process.pid}")
        cls._wait_for_controller()

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
            start_new_session=True,
        )
        print(f"Worker process started with PID: {cls.worker_process.pid}")
        cls.model_name = cls._wait_for_worker_registration()

    @classmethod
    def _wait_for_controller(cls):
        """Wait until the controller is ready to accept worker registration."""
        deadline = time.monotonic() + 30
        last_error = None
        while time.monotonic() < deadline:
            return_code = cls.controller_process.poll()
            if return_code is not None:
                raise RuntimeError(f"controller exited with status {return_code}")

            try:
                response = requests.post("http://localhost:10000/list_models", timeout=5)
                response.raise_for_status()
                return
            except requests.RequestException as error:
                last_error = error
            time.sleep(1)

        raise RuntimeError(f"Controller did not start within 30 seconds: {last_error}")

    @classmethod
    def _wait_for_worker_registration(cls):
        """Wait until the model is loaded and registered with the controller."""
        deadline = time.monotonic() + 300
        last_error = None
        while time.monotonic() < deadline:
            for name, process in (
                ("controller", cls.controller_process),
                ("worker", cls.worker_process),
            ):
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(f"{name} exited with status {return_code}")

            try:
                response = requests.post("http://localhost:10000/list_models", timeout=5)
                response.raise_for_status()
                models = response.json().get("models", [])
                if models:
                    print(f"Worker registered model: {models[0]}")
                    return models[0]
            except (requests.RequestException, ValueError) as error:
                last_error = error
            time.sleep(5)

        raise RuntimeError(f"Worker did not register within 300 seconds: {last_error}")

    @staticmethod
    def _load_test_frame():
        """Extract one JPEG frame from the test video and return it as base64."""
        video_path = os.path.join(os.environ["HOLOHUB_DATA_DIR"], "vila_live", "meeting.mp4")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to extract the VILA test frame")

        frame = subprocess.check_output(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ]
        )
        return base64.b64encode(frame).decode("ascii")

    def test_tinychat(self):
        """Generate a response from the loaded VILA model for a real video frame."""
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful visual AI assistant.\n<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            "Observe the following image: <image>\nDescribe it briefly.<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        request_data = {
            "prompt": prompt,
            "temperature": 0.3,
            "max_tokens": 8,
            "images": [self._load_test_frame()],
            "stop": ["</s>"],
            "n_keep": -1,
            "stream": True,
        }
        response = requests.post(
            "http://localhost:40000/worker_generate_stream",
            data=json.dumps(request_data),
            stream=True,
            timeout=(5, 180),
        )
        response.raise_for_status()

        generated_text = ""
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if not chunk:
                continue
            result = json.loads(chunk.decode())
            self.assertEqual(result["error_code"], 0, result.get("text", ""))
            generated_text = result["text"][len(prompt) :].strip()

        self.assertTrue(generated_text, "VILA returned no generated text")
        print(f"VILA runtime generation succeeded: {generated_text}")

    @classmethod
    def _stop_processes(cls):
        """Stop the TinyChat controller and model worker after running the tests"""
        for name in ("worker_process", "controller_process"):
            process = getattr(cls, name, None)
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                # The process exited after poll() and no longer needs cleanup.
                pass


if __name__ == "__main__":
    unittest.main()
