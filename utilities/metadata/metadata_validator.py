# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import glob
import json
import os
import re
import sys

import jsonschema
from jsonschema import Draft4Validator
from referencing import Registry
from referencing.jsonschema import DRAFT4


def extract_readme_title(readme_path):
    """
    Extract the first markdown title from a README.md file, ignoring any
    leading HTML comment header blocks (e.g. <!-- ... -->) that might be
    present at the top of the file.
    """
    with open(readme_path, "r", encoding="utf-8") as f:
        in_comment_block = False

        for line in f:
            stripped = line.strip()

            # Detect the start of an HTML comment block.
            if stripped.startswith("<!--"):
                # If the comment ends on the same line, just skip it and continue.
                if stripped.endswith("-->"):
                    continue
                in_comment_block = True
                continue

            # If currently inside a comment block, look for its end.
            if in_comment_block:
                if "-->" in stripped:
                    in_comment_block = False
                continue  # Skip all lines while inside comment block.

            # Skip blank lines outside comment blocks.
            if not stripped:
                continue

            # Check for a markdown H1 heading.
            match = re.match(r"^#\s+(.+)$", stripped)
            if match:
                return match.group(1).strip()

    return None


def check_name_matches_readme(metadata_path, json_data):
    """Check if the name in metadata.json matches the title in README.md."""
    # Get the name from metadata.json
    # -----------------------------------------------------------------
    # Currently it only checks for application.
    # However, it can be extended to other entities (operator, benchmark, tutorial, etc.)
    check_entities = ["application"]
    for entity in check_entities:
        if entity in json_data:
            name = json_data[entity].get("name")
            break
    else:
        return True, "Not an application!"

    if name is None:
        return False, "No name field found in metadata.json"

    # Check if the name includes terms like application, holohub and its variations.
    # Note: holoscan is allowed only if it is part of the actual application name, like Isaac Sim
    # Holoscan Bridge.
    forbidden_terms = ["application", "holohub"]
    found_terms = [term for term in forbidden_terms if re.search(term, name, re.IGNORECASE)]
    if found_terms:
        return (
            False,
            f"The 'name' field in metadata.json (\"{name}\") contains "
            f"\"{', '.join(found_terms)}\"."
            f"The name should not include terms like {', '.join(forbidden_terms)}.",
        )

    # Get the title from README.md
    # -----------------------------------------------------------------
    # First check for README.md in the same directory
    metadata_dir = os.path.dirname(metadata_path)
    readme_path = os.path.join(metadata_dir, "README.md")
    if not os.path.exists(readme_path):
        # If no README.md in same directory, check one level up
        readme_path = os.path.join(os.path.dirname(metadata_dir), "README.md")
        if not os.path.exists(readme_path):
            return False, "No README.md found to compare against"

    title = extract_readme_title(readme_path)

    # Check if the name in metadata.json matches the title in README.md
    # -----------------------------------------------------------------
    if name != title:
        return (
            False,
            f"Name in metadata.json ('{name}') does not match README.md title ('{title}' in {readme_path})",
        )

    return True, "Name matches README.md title"


def validate_json(json_data, directory):
    BASE_SCHEMA = "utilities/metadata/project.schema.json"

    # Describe the schema.
    with open(BASE_SCHEMA) as file:
        base_schema = json.load(file)
    registry = Registry().with_resource(base_schema["$id"], DRAFT4.create_resource(base_schema))

    with open(directory + "/metadata.schema.json", "r") as file:
        try:
            execute_api_schema = json.load(file)
        except json.decoder.JSONDecodeError as err:
            return False, err
    validator = Draft4Validator(execute_api_schema, registry=registry)

    try:
        validator.validate(json_data)
    except jsonschema.exceptions.ValidationError as err:
        return False, err

    return True, "valid"


def validate_json_directory(directory, ignore_patterns=[], metadata_is_required: bool = True):
    exit_code = 0
    # Convert json to python object.
    current_wdir = os.getcwd()

    # Check if there is a metadata.json
    subdirs = next(os.walk(current_wdir + "/" + directory))[1]
    for subdir in subdirs:
        ignore = False
        # check if we should ignore the pattern
        for ignore_pattern in ignore_patterns:
            if ignore_pattern in subdir:
                ignore = True

        if ignore is False:
            count = len(
                glob.glob(
                    current_wdir + "/" + directory + "/" + subdir + "/**/metadata.json",
                    recursive=True,
                )
            )
            if count == 0:
                if metadata_is_required:
                    print("ERROR:" + subdir + " does not contain metadata.json file")
                    exit_code = 1
                else:
                    print("WARNING:" + subdir + " does not contain metadata.json file")

    # Check if the metadata is valid
    for name in glob.glob(current_wdir + "/" + directory + "/**/metadata.json", recursive=True):
        ignore = False
        # check if we should ignore the pattern
        for ignore_pattern in ignore_patterns:
            if ignore_pattern in name:
                ignore = True
        if ignore:
            continue

        with open(name, "r") as file:
            try:
                jsonData = json.load(file)
            except json.decoder.JSONDecodeError:
                print("ERROR:" + name + ": invalid")
                exit_code = 1
                continue

            is_valid, msg = validate_json(jsonData, directory)
            if is_valid:
                print(name + ": valid")

                # Check if name matches README title
                # name_matches, name_msg = check_name_matches_readme(name, jsonData)
                # if not name_matches:
                #    print("ERROR:" + name + ": " + name_msg)
                #    exit_code = 1
            else:
                print("ERROR:" + name + ": invalid")
                print(msg)
                exit_code = 1

    return exit_code


# Validate the directories
if __name__ == "__main__":
    exit_code_op = validate_json_directory("operators", ignore_patterns=["template"])
    exit_code_extensions = validate_json_directory("gxf_extensions", ignore_patterns=["utils"])
    exit_code_applications = validate_json_directory("applications", ignore_patterns=["template"])
    exit_code_workflows = validate_json_directory("workflows", ignore_patterns=["template"])
    exit_code_tutorials = validate_json_directory(
        "tutorials", ignore_patterns=["template"], metadata_is_required=False
    )
    exit_code_benchmarks = validate_json_directory("benchmarks")
    exit_code_packages = validate_json_directory("pkg", metadata_is_required=False)

    sys.exit(
        max(
            exit_code_op,
            exit_code_extensions,
            exit_code_applications,
            exit_code_workflows,
            exit_code_tutorials,
            exit_code_benchmarks,
            exit_code_packages,
        )
    )
