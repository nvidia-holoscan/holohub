#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import codecs
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_metadata_files(repo_paths):
    """Recursively search for metadata.json files in the specified repository paths"""
    metadata_files = []

    for repo_path in repo_paths:
        for root, dirs, files in os.walk(repo_path):
            if "metadata.json" in files:
                metadata_files.append(os.path.join(root, "metadata.json"))

    return sorted(metadata_files)


def extract_readme(file_path):
    """Check for the README.md file in the current directory"""
    readme_path = os.path.join(os.path.dirname(file_path), "README.md")
    if os.path.exists(readme_path):
        with codecs.open(readme_path, "r", "utf-8") as readme_file:
            return readme_file.read()
    else:
        # If README.md is not found, look for it one level up
        readme_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), "README.md")
        if os.path.exists(readme_path):
            with codecs.open(readme_path, "r", "utf-8") as readme_file:
                return readme_file.read()
        else:
            return ""


def extract_project_name(metadata_filepath: str) -> str:
    """Extract the project name from the metadata.json file path.

    HoloHub convention is such that a `metadata.json` file
    must be located at either:
    - the named project folder; or
    - a language subfolder one level below the project folder.

    The following are valid examples:
    - applications/my_application/metadata.json -> my_application
    - applications/nested/paths/my_application/cpp/metadata.json -> my_application
    - workflows/my_workflow/metadata.json -> my_workflow

    """
    parts = metadata_filepath.split(os.sep)
    if parts[-2] in ["cpp", "python", "py"]:
        return parts[-3]
    return parts[-2]


def generate_build_and_run_command(metadata: dict) -> str:
    """Generate the build and run command for the application or workflow"""
    language = metadata.get("metadata", {}).get("language", "").lower()
    if language == "python":
        return f'./holohub run {metadata["application_name"]} --language=python'
    elif language in ["cpp", "c++"]:
        return f'./holohub run {metadata["application_name"]} --language=cpp'
    else:
        # Unknown language, use default
        return f'./holohub run {metadata["application_name"]}'


def gather_metadata(repo_paths: list[str], exclude_paths: list[str] = None) -> list[dict]:
    """
    Collect project metadata from JSON files into a single dictionary

    This function will return a list of dictionaries, each containing metadata for a project.

    :input:
        repo_path: str
            The path to the repository to collect metadata from.
        exclude_files: list
            A list of files to exclude from metadata collection.
    :return:
        A list of dictionaries, each containing metadata for a project.
    """
    SCHEMA_TYPES = [
        "application",
        "benchmark",
        "gxf_extension",
        "package",
        "operator",
        "tutorial",
        "workflow",
    ]

    metadata_files = find_metadata_files(repo_paths)
    metadata = []
    exclude_paths = exclude_paths or []

    # Iterate over the found metadata files
    for file_path in metadata_files:
        if any(exclude_path in file_path for exclude_path in exclude_paths):
            continue
        with open(file_path, "r") as file:
            try:
                entries = json.load(file)
                entries = entries if type(entries) is list else [entries]

                for data in entries:
                    try:
                        schema_type = next(key for key in data.keys() if key in SCHEMA_TYPES)
                    except StopIteration:
                        logger.error(
                            'No valid schema type found in metadata file "%s". Available keys: %s',
                            file_path,
                            ", ".join(data.keys()),
                        )
                        continue

                    data["project_type"] = schema_type
                    data["metadata"] = data.pop(schema_type)

                    readme = extract_readme(file_path)
                    project_name = extract_project_name(file_path)
                    source_folder = Path(file_path).parent
                    data["readme"] = readme
                    data["project_name"] = project_name
                    data["source_folder"] = source_folder
                    if source_folder in ["applications", "benchmarks", "workflows"]:
                        data["build_and_run"] = generate_build_and_run_command(data)
                    metadata.append(data)
            except json.decoder.JSONDecodeError as e:
                logger.error('Error parsing JSON file "%s": %s', file_path, e)
                continue

    return metadata


def main(args: argparse.Namespace):
    """Run the gather application"""

    DEFAULT_INCLUDE_PATHS = ["workflows", "applications", "operators", "tutorials"]
    DEFAULT_OUTPUT_FILEPATH = "aggregate_metadata.json"

    repo_paths = args.include or DEFAULT_INCLUDE_PATHS
    output_file = args.output or DEFAULT_OUTPUT_FILEPATH

    metadata = gather_metadata(repo_paths, exclude_files=args.exclude)

    # Write the metadata to the output file
    with open(output_file, "w") as output:
        json.dump(metadata, output, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to collect JSON metadata for HoloHub projects"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Output filepath for JSON collection of project metadata",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        required=False,
        help="Path(s) to search for metadata files",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        required=False,
        help="Filepath(s) to exclude from metadata collection. Takes priority over --include.",
    )
    args = parser.parse_args()
    main(args)
