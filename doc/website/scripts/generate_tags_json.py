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
"""Generate tmp_tag-categories.json for use by the tag-categories.js script.
finding all unique primary categories and their frequencies in the apps.
"""

import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import CATEGORY_ICONS, find_app_pairs, format_category_title, get_git_root, logger

COMPONENT_TYPES = ["applications"]


def generate_tags_json() -> None:
    """Generate tags.json file with metadata about all tags and their associated pages."""
    try:
        git_repo_path = get_git_root()
        logger.info(f"Git repository root: {git_repo_path}")
    except Exception as e:
        logger.error(f"Failed to find Git repository root: {e}")
        logger.error("This script requires Git and must be run from within a Git repository.")
        return

    main_categories = set()
    app_tags = {}

    app_pairs = find_app_pairs(git_repo_path, COMPONENT_TYPES)
    logger.info(f"Found {len(app_pairs)} valid application pairs")

    for app_id, (metadata_path, readme_path) in app_pairs.items():
        try:
            with metadata_path.open("r") as metadata_file:
                metadata = json.load(metadata_file)
            project_type = list(metadata.keys())[0]
            metadata = metadata[project_type]
            tags = metadata.get("tags", [])
            if not tags:
                continue
            app_title = metadata.get("name", "Untitled")
            app_tags[app_title] = tags
            main_categories.add(format_category_title(tags[0]))
        except Exception as e:
            logger.error(f"Failed to process {metadata_path}: {e}")

    category_counts = Counter()
    cat_apps = {}
    for category in main_categories:
        for app_title, tags in app_tags.items():
            if format_category_title(tags[0]) == category:
                category_counts[category] += 1
                if category not in cat_apps:
                    cat_apps[category] = []
                cat_apps[category].append(app_title)

    tag_categories = []
    for category in sorted(main_categories):
        category_data = {
            "title": category,
            "count": category_counts[category],
            "icon": CATEGORY_ICONS[category],
            "ids": cat_apps[category],
        }
        tag_categories.append(category_data)

    output_data_dir = git_repo_path / "doc" / "website" / "docs" / "_data"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    categories_file_path = output_data_dir / "tmp_tag-categories.json"
    with open(categories_file_path, "w") as categories_file:
        json.dump(tag_categories, categories_file, indent=2)
    logger.info(
        f"Generated tag-categories.json with {len(tag_categories)} categories at {categories_file_path}"
    )


def on_startup(**kwargs):
    generate_tags_json()


if __name__ in {"__main__", "<run_path>"}:
    generate_tags_json()
