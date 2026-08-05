# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_api_docs import run_doxygen_and_parse


class RunDoxygenAndParseTest(unittest.TestCase):
    def test_missing_doxygen_stops_api_reference_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            website_dir = Path(temp_dir)
            (website_dir / "Doxyfile").touch()

            with patch("generate_api_docs.shutil.which", return_value=None), self.assertRaisesRegex(
                RuntimeError, "Doxygen is required"
            ):
                run_doxygen_and_parse(website_dir)


if __name__ == "__main__":
    unittest.main()
