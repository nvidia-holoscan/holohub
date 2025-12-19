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
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildHoloscanDB(unittest.TestCase):
    """Test cases for the build_holoscan_db module"""

    def _download_pdf_if_needed(self, pdf_url, pdf_path):
        """Download PDF if it doesn't exist"""
        if os.path.exists(pdf_path):
            return
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        try:
            subprocess.run(
                ["wget", "-nc", "--content-disposition", "-O", pdf_path, pdf_url], check=True
            )
            print(f"Downloaded PDF from {pdf_url} to {pdf_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download PDF from {pdf_url}: {e}")
        except FileNotFoundError:
            raise RuntimeError("wget command not found. Please install wget.")

    def test_pdf_loading_functionality(self):
        """Test that the PDF can be loaded successfully by PyPDFLoader"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_url = "https://developer.nvidia.com/downloads/holoscan-sdk-user-guide"
        pdf_path = os.path.join(current_dir, "docs", "holoscan-sdk-user-guide.pdf")
        self._download_pdf_if_needed(pdf_url, pdf_path)
        self.assertTrue(os.path.exists(pdf_path), f"PDF not found {pdf_path} -- {pdf_url}.")
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            self.assertGreater(len(pages), 10, "PDF should have more than 10 pages")
        except Exception as e:
            self.fail(f"Failed to load PDF: {e}")

    @patch("build_holoscan_db.get_source_chunks")
    @patch("os.walk")
    def test_build_db_with_pdf_integration(self, mock_walk, mock_chunks):
        """Test building database with PDF integration"""
        mock_walk.return_value = []
        from langchain_core.documents import Document

        mock_chunks.return_value = [
            Document(page_content="test chunk", metadata={"source": "test"})
        ]

        with patch("build_holoscan_db.PyPDFLoader") as mock_pdf_loader:
            mock_pages = [
                Mock(page_content="Chapter 1: Introduction to Holoscan"),
                Mock(page_content="Chapter 2: Getting Started"),
            ]
            mock_loader = Mock()
            mock_loader.load_and_split.return_value = mock_pages
            mock_pdf_loader.return_value = mock_loader
            from build_holoscan_db import main

            try:
                main()
                success = True
            except Exception as e:
                success = False
                print(f"Database building failed: {e}")
            self.assertTrue(success, "Database building should succeed when PDF is available")


if __name__ == "__main__":
    unittest.main()
