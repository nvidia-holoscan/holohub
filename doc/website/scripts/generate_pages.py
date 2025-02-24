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
"""Generate the code reference pages and copy Jupyter notebooks and README files."""

import json
import logging
import os
import re
import subprocess
from pathlib import Path

import mkdocs_gen_files

# log stuff
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_last_modified_date(path: str) -> str:
    """Get the last modified date of a file or directory from git."""
    try:
        # Get the last commit date for the path
        cmd = ["git", "log", "-1", "--format=%ad", "--date=short", path]
        result = subprocess.run(cmd, cwd=path, capture_output=True, text=True, check=True)
        # Convert YYYY-MM-DD to Month DD, YYYY format
        date = result.stdout.strip()
        try:
            year, month, day = date.split("-")
            months = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            return f"{months[int(month)-1]} {int(day)}, {year}"
        except subprocess.CalledProcessError:
            return date
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e}")
        logger.error(f"Git error output: {e.stderr}")
        return "Unknown"


def parse_metadata_file(metadata_file: Path, statistics) -> None:
    """Copy README file from a sub-package to the user guide's developer guide directory.

    Returns:
        None
    """
    # Disable application with {{ in the name
    if "{{" in str(metadata_file):
        return

    # Read the file
    # Parse the JSON data
    with open(metadata_file, "r") as metadatafile:
        metadata = json.load(metadatafile)
        key = list(metadata.keys())[0]
        dest_dir = str(key) + "s"

    # Extract the application name, removing cpp/python from the path for counting
    if dest_dir == "applications":
        path = re.sub(r".*/applications/", "", str(metadata_file)).removesuffix("/metadata.json")
        base_path = re.sub(r"/(cpp|python)$", "", path)
        statistics["unique_applications"].add(base_path)
    elif dest_dir == "workflows":
        path = re.sub(r".*/workflows/", "", str(metadata_file)).removesuffix("/metadata.json")
        base_path = re.sub(r"/(cpp|python)$", "", path)
        statistics["unique_workflows"].add(base_path)
    elif dest_dir == "operators":
        path = str(metadata_file).removesuffix("/metadata.json")
        path = path.split("/")[-1]
        base_path = re.sub(r"/(cpp|python)$", "", path)
        statistics["unique_operators"].add(base_path)
    elif dest_dir == "tutorials":
        path = str(metadata_file).removesuffix("/metadata.json")
        path = path.split("/")[-1]
        base_path = re.sub(r"/(cpp|python)$", "", path)
        statistics["unique_tutorials"].add(base_path)
    elif dest_dir == "benchmarks":
        path = str(metadata_file).removesuffix("/metadata.json")
        path = path.split("/")[-1]
        base_path = re.sub(r"/(cpp|python)$", "", path)
        statistics["unique_benchmarks"].add(base_path)
    else:
        logger.error(f"Don't know the output path for: {dest_dir}")
        return

    # Extract the "tags" into a Python list
    tags = metadata[key]["tags"]
    name = metadata[key]["name"]
    platforms = metadata[key]["platforms"]
    authors = metadata[key]["authors"]
    if "language" in metadata[key]:
        language = metadata[key]["language"]
    version = metadata[key]["version"]
    minimum_required_version = metadata[key]["holoscan_sdk"]["minimum_required_version"]
    tested_versions = metadata[key]["holoscan_sdk"]["tested_versions"]
    metric = metadata[key]["ranking"]

    metric_str = "Level 0 - Core Stable"
    if metric == 1:
        metric_str = "Level 1 - Highly Reliable"
    if metric == 2:
        metric_str = "Level 2 - Trusted"
    if metric == 3:
        metric_str = "Level 3 - Developmental"
    if metric == 4:
        metric_str = "Level 4 - Experimental"
    if metric == 5:
        metric_str = "Level 5 - Obsolete"

    output_text = "---"
    output_text += "\ntags:"
    for tag in tags:
        output_text += f"\n - {tag}"
    output_text += f"\ntitle: {name}"
    dir = path.split("/")[-1]
    if dir == "python":
        output_text += " (Python)"
    elif dir == "cpp":
        output_text += " (C++)"
    output_text += "\n---\n"

    # Finds the README.md
    readme_path = str(metadata_file).replace("metadata.json", "README.md")
    readme_parent = 0
    if not os.path.exists(readme_path):
        readme_path = str(metadata_file.parent) + "/README.md"
        readme_parent = 1
    if not os.path.exists(readme_path):
        readme_path = str(metadata_file.parent.parent) + "/README.md"
        readme_parent = 2

    if os.path.exists(readme_path):
        with open(readme_path, "r") as readme_file:
            readme_text = readme_file.read()

            # Regular expression pattern to match paths containing .gif, .png, or .jpg
            pattern = r'["(\[][^:")]*\.(?:gif|png|jpg)[")\]]'

            # Find all matches in the string
            matches = re.findall(pattern, readme_text)

            # Tune the path
            relative_path = path
            if readme_parent == 2:
                relative_path = os.path.dirname(path)

            # Print the matches
            for match in matches:

                # Find the URL inside [](image)
                parenthensis_match = re.search(r"\((.*?)\)", match)
                if parenthensis_match:
                    match = parenthensis_match.group(1)

                match = match.strip('"()[]')
                imgmatch = match
                if match.startswith("."):
                    imgmatch = match[1:]
                if imgmatch.startswith("/"):
                    imgmatch = imgmatch[1:]
                readme_text = readme_text.replace(
                    match,
                    "https://github.com/nvidia-holoscan/holohub/blob/main/"
                    + dest_dir
                    + "/"
                    + relative_path
                    + "/"
                    + imgmatch
                    + "?raw=true",
                )

            # Find the first heading
            pattern = r"^#\s+(.+)"
            match = re.match(pattern, readme_text)

            header_text = (
                ":octicons-person-24: **Authors:** "
                + ", ".join(f"{author['name']} ({author['affiliation']})" for author in authors)
                + "<br>"
            )
            header_text += (
                ":octicons-device-desktop-24: **Supported platforms:** "
                + ", ".join(platforms)
                + "<br>"
            )

            # Add last modified date from git
            last_modified = get_last_modified_date(str(metadata_file.parent))
            header_text += f":octicons-clock-24: **Last modified:** {last_modified}<br>"

            if "language" in locals():
                header_text += ":octicons-code-square-24: **Language:** " + language + "<br>"
            header_text += ":octicons-tag-24: **Latest version:** " + version + "<br>"
            header_text += (
                ":octicons-stack-24: **Minimum Holoscan SDK version:** "
                + minimum_required_version
                + "<br>"
            )
            header_text += (
                ":octicons-beaker-24: **Tested Holoscan SDK versions:** "
                + ", ".join(tested_versions)
                + "<br>"
            )
            header_text += (
                ":octicons-sparkle-fill-24: **Contribution metric:** " + metric_str + "<br>"
            )

            # Add the header text
            if match:
                title = match.group(1)
                title = (
                    "["
                    + title
                    + "](https://github.com/nvidia-holoscan/holohub/tree/main/"
                    + dest_dir
                    + "/"
                    + path
                    + ")"
                )
                output_text += readme_text.replace(match.group(1), title + "\n" + header_text, 1)
            else:
                output_text += readme_text

    dest_directory = dest_dir + "/" + path
    dest_file = dest_directory + ".md"

    with mkdocs_gen_files.open(dest_file, "w") as fd:
        fd.write(output_text)


def generate_pages() -> None:
    """Generate pages for documentation.

    This function orchestrates the entire process of generating API references,
    copying README files for workflow, applications and operators.

    Returns:
        None
    """
    root = Path(__file__).parent.parent.parent.parent

    statistics = {
        "unique_operators": set(),
        "unique_tutorials": set(),
        "unique_applications": set(),
        "unique_workflows": set(),
        "unique_benchmarks": set(),
    }

    logger.info(f"root: {root}")

    # Parse the metadata.json files
    for metadata_file in root.rglob("metadata.json"):
        parse_metadata_file(metadata_file, statistics)

    logger.info(f"Stats: {statistics}")
    # Write the home page
    homefile_path = str(Path(__file__).parent.parent) + "/docs/index.md"
    with open(homefile_path, "r") as home_file:
        home_text = home_file.read()
        home_text = home_text.replace("#operators", str(len(statistics["unique_operators"])))
        home_text = home_text.replace("#tutorials", str(len(statistics["unique_tutorials"])))
        home_text = home_text.replace("#applications", str(len(statistics["unique_applications"])))
        home_text = home_text.replace("#workflows", str(len(statistics["unique_workflows"])))
        home_text = home_text.replace("#benchmarks", str(len(statistics["unique_benchmarks"])))

    with mkdocs_gen_files.open("index.md", "w") as fd:
        fd.write(home_text)

if __name__ in {"__main__", "<run_path>"}:
    # Check if name is either '__main__', or the equivalent default in `runpy.run_path(...)`, which is '<run_path>'
    generate_pages()
