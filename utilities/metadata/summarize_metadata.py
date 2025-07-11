# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import semver
from gather_metadata import gather_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__file__)

DEFAULT_DETAIL_COLUMNS = [
    "name",
    "project_type",
    "platforms",
    "language",
    "ranking",
    "holoscan_sdk.tested_versions",
]
DEFAULT_SORT_COLUMNS = ["project_type", "name"]


class ProjectType(Enum):
    """Types of subprojects managed in the HoloHub repository that define metadata schemas."""

    APPLICATION = 0
    GXF_EXTENSION = 1
    OPERATOR = 2
    WORKFLOW = 3


@dataclass
class ProjectTypeData:
    """HoloHub repository information related to each subproject schema."""

    project_type: ProjectType
    folder_name: str
    schema_name: str

    @property
    def schema_filepath(self) -> str:
        return f"{self.folder_name}/metadata.schema.json"

    @property
    def schema(self) -> dict:
        if not self.schema:
            with open(self.schema_filepath, "r") as file:
                self.schema = json.load(file)
        return self.schema


project_type_data = [
    ProjectTypeData(ProjectType.APPLICATION, "applications", "application"),
    ProjectTypeData(ProjectType.GXF_EXTENSION, "gxf_extensions", "gxf_extension"),
    ProjectTypeData(ProjectType.OPERATOR, "operators", "operator"),
    ProjectTypeData(ProjectType.WORKFLOW, "workflows", "workflow"),
]


def collect_metadata() -> pd.DataFrame:
    """Gather HoloHub project metadata into a DataFrame"""
    METADATA_DIRECTORIES = ["applications", "workflows", "gxf_extensions", "operators"]

    # Ingest project metadata files
    metadata = gather_metadata(METADATA_DIRECTORIES)
    for entry in metadata:
        entry["metadata"]["project_type"] = entry["source_folder"]
    frames = [pd.json_normalize(entry["metadata"]) for entry in metadata]

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(by=DEFAULT_SORT_COLUMNS)
        .reset_index(drop=True)
    )


def process_versions(metadata_df: pd.DataFrame, key="holoscan_sdk.tested_versions") -> dict:
    """
    Summarize version information from a dataframe as a frequency table.

    Expects semantic versions (major.minor.revision or major.minor)
    """
    freq = defaultdict(int)
    for val in metadata_df[key].dropna():
        versions = val if isinstance(val, list) else [val]
        for version_str in versions:
            try:
                version = semver.Version.parse(version_str)
            except ValueError:
                version = semver.Version.parse(version_str + ".0")
            freq[str(version)] += 1
    return pd.DataFrame(freq, index=[0])


def summarize_subprojects(metadata: pd.DataFrame) -> str:
    """Report summary statistics for ingested subproject metadata"""
    holoscan_versions = process_versions(metadata, "holoscan_sdk.tested_versions")
    gxf_versions = process_versions(metadata, "gxf_version.tested_versions")

    summary = "######################### HoloHub Metadata Summary #########################\n\n"

    for key in ["project_type", "language", "ranking"]:
        summary += metadata[key].value_counts().to_string()
        summary += "\n\n"
    summary += "Holoscan tested versions:\n"
    for version in holoscan_versions:
        summary += f"<{version}>: {holoscan_versions[version][0]}\n"
    summary += "\n\n"
    summary += "GXF tested versions:\n"
    for version in gxf_versions:
        summary += f"<{version}>: {gxf_versions[version][0]}\n"
    summary += "\n\n"
    return summary


def main(args: argparse.Namespace):
    """Ingest and output metadata"""
    if args.output and not args.output.endswith(".csv"):
        logger.warning(
            f"Expected CSV output path but received input {args.output}, proceeding anyway"
        )
    if args.markdown and not args.markdown.endswith(".md"):
        logger.warning(
            f"Expected markdown file (.md) but received input {args.markdown}, proceeding anyway"
        )

    metadata_df = collect_metadata()
    if args.output:
        metadata_df.to_csv(args.output, index=False)

    metadata_summary_df = metadata_df[DEFAULT_DETAIL_COLUMNS]

    if not args.quiet:
        logger.info(summarize_subprojects(metadata_df))
        logger.info(
            "######################### HoloHub Subprojects Summary #########################\n\n"
        )
        logger.info(metadata_summary_df.to_string(index=False))

    if args.markdown:
        metadata_summary_df.to_markdown(args.markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to collect and inspect metadata for HoloHub projects"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Output filepath for comma-delimited matrix of project metadata",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        required=False,
        help="Output filepath for abbreviated project fields in markdown format",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        required=False,
        action="store_true",
        help="Run quietly without console output",
    )
    args = parser.parse_args()
    main(args)
