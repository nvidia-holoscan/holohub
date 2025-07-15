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

import json
import logging
import time
import urllib

import requests


class VLM:
    def __init__(self):
        self.llm_url = "http://0.0.0.0:40000"
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def generate_response(self, user_prompt, image_b64):
        """
        Stream a response from the LLM and optionally write to the output queue.
        This method will attempt to connect to the LLM server for up to 30 seconds.
        """
        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful visual AI assistant.\n"
        prompt += "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
        prompt += f"Observe the following image: <image>\nRespond according to the following prompt: {user_prompt}<|eot_id|>\n"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        start_time = time.time()
        # Attempt making the request for up to 60 seconds
        while time.time() - start_time < 60:
            try:
                request_data = {
                    "prompt": prompt,
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "images": [image_b64],
                    "stop": ["</s>"],
                    "n_keep": -1,
                    "stream": True,
                }
                resData = requests.request(
                    "POST",
                    urllib.parse.urljoin(self.llm_url, "/worker_generate_stream"),
                    data=json.dumps(request_data),
                    stream=True,
                )
                response = ""
                for chunk in resData.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode())
                        if data["error_code"] == 0:
                            output = data["text"][len(prompt) :].strip()
                            yield output
                return response
            except Exception:
                self._logger.debug("Failed connection to VLM server, retrying in 5 seconds...")
                time.sleep(5)
        raise ConnectionError("VILA server unavailable after 60 seconds")
