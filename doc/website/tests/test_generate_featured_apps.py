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
    def test_component_title_link_does_not_trigger_card_click(self):
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
                '<a href="/holohub/applications/bci_visualization/" '
                'onclick="event.stopPropagation();">'
                "Kernel Flow BCI Real-Time Visualization</a>",
                card_html,
            )

    def test_language_specific_component_links_to_its_generated_page(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            git_repo_path = Path(temp_dir)
            component_dir = git_repo_path / "operators" / "velodyne_lidar" / "cpp"
            component_dir.mkdir(parents=True)
            (component_dir / "README.md").write_text("# VelodyneLidarOp\n", encoding="utf-8")
            metadata_path = component_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "operator": {
                            "name": "VelodyneLidarOp",
                            "description": "Convert Velodyne lidar packets.",
                            "language": "C++",
                            "tags": ["Sensor", "Lidar"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            card_html = generate_featured_component_card(metadata_path, git_repo_path)

            self.assertIn(
                'href="/holohub/operators/velodyne_lidar/cpp/"',
                card_html,
            )


if __name__ == "__main__":
    unittest.main()
