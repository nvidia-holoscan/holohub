#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_metadata_files(repo_paths):
    """Recursively search for metadata.json files in the specified repository paths"""
    metadata_files = []

    for repo_path in repo_paths:
        for root, dirs, files in os.walk(repo_path):
            if "metadata.json" in files:
                metadata_files.append(os.path.join(root, "metadata.json"))

    return metadata_files


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


def extract_application_name(metadata_filepath: str) -> str:
    """Extract the application name from the README file path.

    HoloHub convention is such that an application `metadata.json` file
    must be located at either:
    - the named application project folder; or
    - a language subfolder one level below the application project folder.

    The following are valid examples:
    - applications/my_application/metadata.json -> my_application
    - applications/nested/paths/my_application/cpp/metadata.json -> my_application
    """
    parts = metadata_filepath.split(os.sep)
    if parts[-2] in ["cpp", "python"]:
        return parts[-3]
    return parts[-2]


def generate_build_and_run_command(metadata: dict) -> str:
    """Generate the build and run command for the application"""
    language = metadata.get("metadata", {}).get("language", "").lower()
    if language == "python":
        return f'./dev_container build_and_run {metadata["application_name"]} --language python'
    elif language in ["cpp", "c++"]:
        return f'./dev_container build_and_run {metadata["application_name"]} --language cpp'
    else:
        # Unknown language, use default
        return f'./dev_container build_and_run {metadata["application_name"]}'


def gather_metadata(repo_path, exclude_files: None) -> dict:
    """Collect project metadata from JSON files into a single dictionary"""
    SCHEMA_TYPES = ["application", "operator", "gxf_extension", "tutorial"]

    metadata_files = find_metadata_files(repo_path)
    metadata = []
    exclude_files = exclude_files or []

    # Iterate over the found metadata files
    for file_path in metadata_files:
        if file_path in exclude_files:
            continue
        with open(file_path, "r") as file:
            try:
                data = json.load(file)

                for schema_type in SCHEMA_TYPES:
                    if schema_type in data:
                        data["metadata"] = data.pop(schema_type)
                        break

                readme = extract_readme(file_path)
                application_name = extract_application_name(file_path)
                source_folder = os.path.normpath(file_path).split("/")[0]
                data["readme"] = readme
                data["application_name"] = application_name
                data["source_folder"] = source_folder
                if source_folder == "applications":
                    data["build_and_run"] = generate_build_and_run_command(data)
                metadata.append(data)
            except json.decoder.JSONDecodeError as e:
                logger.error('Error parsing JSON file "%s": %s', file_path, e)
                continue

    return metadata


def main(args: argparse.Namespace):
    """Run the gather application"""

    DEFAULT_INCLUDE_PATHS = ["applications", "operators", "tutorials"]
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
