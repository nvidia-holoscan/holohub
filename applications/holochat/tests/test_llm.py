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
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LLM


class TestLLM(unittest.TestCase):
    """Test cases for the LLM module"""

    def setUp(self):
        self.server_path = "/workspace/llama.cpp/build/bin/llama-server"
        self.timeout = 10  # 10 second timeout

    def test_llm_initialization(self):
        """Test LLM initialization"""
        # os.environ is a dict-like mapping, not an object with attributes, so use patch.dict.
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "DUMMY_FOR_TESTS"}, clear=False):
            llm = LLM(is_mcp=False, is_local=False)
            self.assertIsNotNone(llm)
            self.assertEqual(llm.config.max_prompt_tokens, 3000)
            self.assertEqual(llm.config.nim_url, "https://integrate.api.nvidia.com/v1")
            self.assertEqual(llm.config.nim_model, "meta/llama-3.3-70b-instruct")

    def test_token_calculation(self):
        """Test token usage calculation using actual LLM class"""
        llm = LLM(is_mcp=True)
        test_text = "test text"
        token_count = llm.calculate_token_usage(test_text)
        self.assertEqual(token_count, 2)
        empty_count = llm.calculate_token_usage("")
        self.assertEqual(empty_count, 0)

    def test_server_initialization(self):
        """Test MCP server initialization"""
        from mcp_server import HoloscanContextServer

        mock_config = SimpleNamespace(
            mcp_server_name="test-server",
            default_num_docs=5,
            max_num_docs=20,
            search_threshold=0.35,
        )
        mock_db = Mock()
        server = HoloscanContextServer(mock_config, mock_db)
        self.assertIsNotNone(server)
        self.assertEqual(server.config.mcp_server_name, "test-server")
        self.assertEqual(server.db, mock_db)

    def test_server_can_start(self):
        """Test that the llama.cpp server command can start successfully."""
        result = subprocess.run(
            [self.server_path, "--help"], capture_output=True, text=True, timeout=self.timeout
        )
        self.assertIn("usage", result.stdout)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
