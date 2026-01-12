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

import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestChatbot(unittest.TestCase):
    """Test cases for the chatbot module"""

    def test_parse_args_default(self):
        """Test argument parsing with default values"""
        from chatbot import parse_args

        with patch("sys.argv", ["chatbot.py"]):
            args = parse_args()
            self.assertFalse(args.local)
            self.assertFalse(args.mcp)

        with patch("sys.argv", ["chatbot.py", "--local"]):
            args = parse_args()
            self.assertTrue(args.local)
            self.assertFalse(args.mcp)

    def test_ask_question_function(self):
        """Test ask_question function"""
        from chatbot import ask_question

        message = "Hello"
        chat_history = None
        result_message, result_history = ask_question(message, chat_history)
        self.assertEqual(result_message, "")
        # Gradio 6.x "messages" format: list of {"role": ..., "content": ...}
        self.assertEqual(len(result_history), 2)
        self.assertEqual(result_history[0]["role"], "assistant")
        self.assertIn("Welcome to HoloChat", result_history[0]["content"])
        self.assertEqual(result_history[1]["role"], "user")
        self.assertEqual(result_history[1]["content"], message)

    @patch("chatbot.start_mcp_server")
    @patch("chatbot.LLM")
    def test_main_mcp_mode(self, mock_llm, mock_start_mcp_server):
        """Test main function in MCP mode"""
        mock_llm_instance = Mock()
        mock_llm_instance.config = Mock()
        mock_llm_instance.db = Mock()
        mock_llm.return_value = mock_llm_instance

        with patch("chatbot.args") as mock_args:
            mock_args.mcp = True
            mock_args.local = False

            from chatbot import main

            main()
            mock_llm.assert_called_once_with(is_local=False, is_mcp=True)
            mock_start_mcp_server.assert_called_once_with(
                mock_llm_instance.config, mock_llm_instance.db
            )


if __name__ == "__main__":
    unittest.main()
