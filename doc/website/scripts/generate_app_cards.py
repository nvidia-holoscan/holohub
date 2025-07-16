#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the app cards JSON data for use in the applications by category page."""

import json
import os
import sys
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(script_dir)) if str(script_dir) not in sys.path else None

from common_utils import (  # noqa: E402
    extract_first_sentences,
    extract_image_from_readme,
    find_app_pairs,
    format_category_title,
    get_full_image_url,
    get_git_root,
    logger,
    parse_metadata_file,
)

OUTPUT_FILE = "doc/website/docs/_data/tmp_app_cards.json"
COMPONENT_TYPES = ["applications"]


def get_app_url(readme_path, git_repo_path):
    """Generate the relative path for the app documentation link based on README location."""
    if not readme_path:
        return None
    readme_dir = readme_path.parent
    rel_path = readme_dir.relative_to(git_repo_path)
    return f"{rel_path}/"


def generate_app_cards():
    """Generate app cards data for all applications."""
    git_repo_path = get_git_root()

    # Find all valid app pairs (metadata.json and README.md)
    app_pairs = find_app_pairs(git_repo_path, COMPONENT_TYPES)
    logger.info(f"Found {len(app_pairs)} valid application pairs")

    app_cards = {}

    # Process each application pair
    for app_id, (metadata_path, readme_path) in app_pairs.items():
        app_name = app_id.split("/")[-1]
        logger.info(f"Processing app: {app_name} from {app_id}")

        # Parse metadata
        metadata, _ = parse_metadata_file(metadata_path)

        readme_content = None
        try:
            with open(readme_path, "r") as f:
                readme_content = f.read()
        except Exception as e:
            logger.error(f"Error reading README for {app_name}: {e}")

        # Extract description
        description = None
        if metadata and "description" in metadata:
            description = metadata["description"]
        elif readme_content:
            description = extract_first_sentences(readme_content)

        if not description:
            description = "(No description available.)"

        # Extract image
        image_url = None
        if readme_content:
            image_path = extract_image_from_readme(readme_content)
            if image_path:
                image_url = get_full_image_url(image_path, readme_path)
                logger.info(f"Found image for {app_name}: {image_url}")

        proper_name = metadata.get("name", app_name) if metadata else app_name
        app_url = get_app_url(readme_path, git_repo_path)
        mapped_tags = [format_category_title(tag) for tag in metadata.get("tags", [])]
        # Create app card data
        app_cards[proper_name] = {
            "name": proper_name,
            "description": description,
            "image_url": image_url,
            "tags": mapped_tags,
            "app_title": proper_name,
            "app_url": app_url,
        }

    # Write the JSON file
    output_path = git_repo_path / OUTPUT_FILE
    with open(output_path, "w") as f:
        json.dump(app_cards, f, indent=2)

    logger.info(f"Generated app cards data for {len(app_cards)} applications")
    return app_cards


def on_startup(**kwargs):
    generate_app_cards()


if __name__ in {"__main__", "<run_path>"}:
    generate_app_cards()
