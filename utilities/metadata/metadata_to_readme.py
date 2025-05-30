#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import re
import sys
from pathlib import Path


def find_first_section_heading(readme_content):
    """
    Find the position of the first section heading (## ...) in the README.
    Returns the title and its position in the content.
    """
    lines = readme_content.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^## ", line.strip()):
            return line.strip(), i

    return None, -1


def find_insertion_point(readme_content):
    """
    Find the best position to insert the Quick Run section.
    Returns the position and whether it should be inserted after that line.
    """
    lines = readme_content.split("\n")

    # Priority order for sections to look for
    section_patterns = [
        r"^## Requirement",
        r"^## Dependencies",
        r"^## Setup",
        r"^## Installation",
        r"^## Usage",
    ]

    # First try to find a common section heading
    for pattern in section_patterns:
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip(), re.IGNORECASE):
                return i, False  # Insert before this section

    # If no common section found, find the first section heading
    first_section, first_section_pos = find_first_section_heading(readme_content)
    if first_section_pos != -1:
        return first_section_pos, False  # Insert before the first section

    # If no section headings, look for the end of the introduction
    # (typically after 1-3 paragraphs with no empty lines between them)
    intro_end = -1
    paragraph_count = 0

    for i, line in enumerate(lines):
        if not line.strip():  # Empty line
            if i > 0 and lines[i - 1].strip():  # Previous line wasn't empty
                paragraph_count += 1

                # If we've seen 2-3 paragraphs, this is a good spot
                if paragraph_count >= 2:
                    intro_end = i
                    break

    if intro_end != -1:
        return intro_end, True  # Insert after this line (which is empty)

    # Fallback: insert after the title (assumed to be the first line)
    return 1, True


def generate_simple_quick_run_section(metadata_json):
    """
    Generate a simple Quick Run section based on metadata.json that includes all launch commands.
    """
    if "launch" not in metadata_json["application"]:
        return None

    launch_data = metadata_json["application"]["launch"]
    if not launch_data:
        return None

    # Get the default command (first item in the launch array)
    default_command = launch_data[0].get("command", "")
    if not default_command:
        return None

    content = ["## Quick Run\n"]
    content.append(
        "<!-- This section is automatically generated from metadata.json. "
        "Do not modify manually. Update the metadata.json file instead. -->"
    )
    content.append("Run this application with:\n")
    content.append("```bash")
    content.append(default_command)
    content.append("```\n")

    # Only add alternative commands if there are more than one
    if len(launch_data) > 1:
        content.append("### Additional Run Options\n")

        for i, launch_option in enumerate(launch_data[1:], 1):
            desc = launch_option.get("description", f"Option {i}")
            cmd = launch_option.get("command", "")

            content.append(f"**{desc}**:")
            content.append("```bash")
            content.append(cmd)
            content.append("```")

            # Add note about hardware requirements if present
            hardware_req = launch_option.get("requires_hardware", [])
            if hardware_req:
                content.append(f"*Note: Requires {', '.join(hardware_req)} hardware.*\n")
            else:
                content.append("")  # Add empty line for spacing

    return "\n".join(content)


def update_quick_run_in_readme(readme_content, quick_run_content):
    """
    Update the Quick Run section in the README.
    """
    if "## Quick Run" in readme_content:
        quick_run_start = readme_content.find("## Quick Run")

        next_section_match = re.search(r"\n## [^\n]+", readme_content[quick_run_start + 1 :])
        if next_section_match:
            next_section_start = quick_run_start + 1 + next_section_match.start()
            updated_content = (
                readme_content[:quick_run_start]
                + quick_run_content
                + readme_content[next_section_start:]
            )
        else:
            updated_content = readme_content[:quick_run_start] + quick_run_content
    else:
        insertion_pos, insert_after = find_insertion_point(readme_content)

        if insertion_pos != -1:
            lines = readme_content.split("\n")
            if insert_after:
                lines.insert(insertion_pos + 1, "\n" + quick_run_content)
            else:
                lines.insert(insertion_pos, quick_run_content + "\n")

            updated_content = "\n".join(lines)
        else:
            updated_content = readme_content + "\n\n" + quick_run_content

    return updated_content


def update_readme_with_launch_commands(readme_path, metadata_path):
    """
    Update README.md with an integrated Quick Run section from metadata.json.
    """
    # Load metadata.json
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading metadata.json: {e}")
        return False

    # Generate Quick Run section with integrated launch commands
    quick_run_content = generate_simple_quick_run_section(metadata)

    if not quick_run_content:
        print(f"No launch commands found in {metadata_path}")
        return False

    # Load README.md
    try:
        with open(readme_path, "r") as f:
            readme_content = f.read()
    except FileNotFoundError:
        print(f"README.md not found at {readme_path}")
        return False

    # Update the README with the integrated Quick Run section
    updated_content = update_quick_run_in_readme(readme_content, quick_run_content)

    # Write updated content back to README.md
    with open(readme_path, "w") as f:
        f.write(updated_content)

    print(
        f"Successfully updated {readme_path} with integrated Quick Run section from {metadata_path}"
    )
    return True


def process_all_projects(base_path, project_type=None):
    """
    Process all projects of a given type (applications, operators, etc.) or all types if none specified.
    """
    base_path = Path(base_path)
    project_types = (
        ["applications", "operators", "workflows", "benchmarks"]
        if not project_type
        else [project_type]
    )

    total_updated = 0
    total_failed = 0

    for project_type in project_types:
        project_path = base_path / project_type
        if not project_path.exists():
            print(f"Project type directory '{project_type}' not found.")
            continue

        print(f"Processing {project_type}...")

        # Find all metadata.json files
        metadata_files = list(project_path.glob("**/metadata.json"))

        for metadata_file in metadata_files:
            project_dir = metadata_file.parent
            readme_file = project_dir / "README.md"

            print(f"Processing {metadata_file.relative_to(base_path)}...")

            if not readme_file.exists():
                print(f"WARNING: No README.md found in {project_dir.relative_to(base_path)}")
                total_failed += 1
                continue

            success = update_readme_with_launch_commands(readme_file, metadata_file)
            if success:
                total_updated += 1
            else:
                total_failed += 1

    print(f"\nSummary: {total_updated} READMEs updated, {total_failed} failed.")
    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate launch commands from metadata.json into README.md files"
    )
    parser.add_argument(
        "--project-type", "-t", help="Project type to process (applications, operators, etc.)"
    )
    parser.add_argument("--single-project", "-p", help="Process a single project directory")
    parser.add_argument(
        "--base-path",
        default=".",
        help="Base path for HoloHub repository (default: current directory)",
    )

    args = parser.parse_args()

    if args.single_project:
        # Process a single project
        project_path = Path(args.base_path) / args.single_project
        metadata_path = project_path / "metadata.json"
        readme_path = project_path / "README.md"

        if not metadata_path.exists():
            print(f"ERROR: metadata.json not found in {project_path}")
            return 1

        if not readme_path.exists():
            print(f"ERROR: README.md not found in {project_path}")
            return 1

        success = update_readme_with_launch_commands(readme_path, metadata_path)
        return 0 if success else 1
    else:
        # Process all projects of a given type
        success = process_all_projects(args.base_path, args.project_type)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
