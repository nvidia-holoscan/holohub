# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_featured_apps import generate_featured_component_card


class GenerateFeaturedComponentCardTest(unittest.TestCase):
    def test_component_title_is_a_crawlable_link(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            git_repo_path = Path(temp_dir)
            component_dir = git_repo_path / "applications" / "bci_visualization"
            component_dir.mkdir(parents=True)
            metadata_path = component_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "application": {
                            "name": "Kernel Flow BCI Real-Time Visualization",
                            "description": "A brain-computer interface application for Holoscan.",
                            "language": "Python",
                            "tags": ["Visualization", "BCI"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            card_html = generate_featured_component_card(metadata_path, git_repo_path)

            self.assertIn(
                '<a href="/holohub/applications/bci_visualization/">'
                "Kernel Flow BCI Real-Time Visualization</a>",
                card_html,
            )


if __name__ == "__main__":
    unittest.main()
