# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_module_pages import build_metadata_header, generate_module_card

ENTRY = {"name": "holoscan-example", "nvidia_quality_score": 3}


def _module(**fields):
    return {"name": "holoscan-example", "description": "Example module.", **fields}


class ModuleCardLayoutTest(unittest.TestCase):
    def test_quality_score_label(self):
        header = build_metadata_header(_module(), ENTRY)
        self.assertIn("**Quality score:**", header)
        self.assertNotIn("NVIDIA quality score", header)
        self.assertIn('title="Quality score: 3/5"', generate_module_card(ENTRY, _module()))

    def test_long_description_is_clamped_with_full_text_tooltip(self):
        description = "Long description " * 20
        card = generate_module_card(ENTRY, _module(description=description))
        self.assertIn("-webkit-line-clamp:3", card)
        self.assertIn(f'title="{description}"', card)


if __name__ == "__main__":
    unittest.main()
