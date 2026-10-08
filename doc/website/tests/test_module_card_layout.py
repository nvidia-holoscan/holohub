# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import sys
import unittest
import unittest.mock
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_module_pages
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

    def test_detail_page_shows_full_description(self):
        description = " ".join(f"word{i}" for i in range(60))
        page = io.StringIO()
        gen_files = unittest.mock.MagicMock()
        gen_files.open.return_value.__enter__.return_value = page
        with unittest.mock.patch.object(generate_module_pages, "mkdocs_gen_files", gen_files):
            generate_module_pages.generate_detail_page(
                ENTRY, _module(description=description), "# Example\n\nBody.\n", "", None
            )
        body = page.getvalue().split("---\n", 2)[2]
        self.assertIn(f'<p class="module-description">{description}</p>', body)
        self.assertLess(body.index("module-description"), body.index("Quality score"))


if __name__ == "__main__":
    unittest.main()
