# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Utility helpers shared across metadata consumers.

import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

METADATA_DIRECTORY_CONFIG = {
    "applications": {"ignore_patterns": ["template"], "metadata_is_required": True},
    "benchmarks": {},
    "gxf_extensions": {"ignore_patterns": ["utils"], "metadata_is_required": True},
    "operators": {"ignore_patterns": ["template"], "metadata_is_required": True},
    "pkg": {"metadata_is_required": False},
    "tutorials": {"ignore_patterns": ["template"], "metadata_is_required": False},
    "workflows": {"ignore_patterns": ["template"], "metadata_is_required": True},
}

DEFAULT_INCLUDE_PATHS = tuple(METADATA_DIRECTORY_CONFIG.keys())


def normalize_language(language: str | None, *, strict: bool = False) -> str:
    """Normalize language names, optionally enforcing known HoloHub languages."""
    if not language or not isinstance(language, str):
        return ""
    lang = language.strip().lower()
    if lang in ("cpp", "c++"):
        normalized = "cpp"
    elif lang in ("python", "py"):
        normalized = "python"
    else:
        normalized = lang

    if strict and normalized not in ("", "cpp", "python"):
        raise ValueError(f"Invalid language: {language}")
    return normalized


def list_normalized_languages(language, *, strict: bool = False) -> list[str]:
    """Return a list of normalized language tags from a single value or sequence."""
    if isinstance(language, str) or language is None:
        values = [language]
    elif isinstance(language, Iterable):
        values = list(language)
    else:
        values = []

    normalized = [
        normalize_language(value, strict=strict)
        for value in values
        if value is None or isinstance(value, str)
    ]
    normalized = [value for value in normalized if value]
    return normalized or [""]


def iter_metadata_paths(
    repo_paths: Sequence[str | os.PathLike],
    *,
    exclude_patterns: Sequence[str] | None = None,
) -> Iterator[str]:
    """Yield filtered metadata.json paths."""
    excludes = [pattern for pattern in (exclude_patterns or []) if pattern]

    for repo_path in repo_paths:
        for root, _, files in os.walk(repo_path):
            if "metadata.json" not in files:
                continue
            file_path = os.path.join(root, "metadata.json")
            if excludes and any(pattern in file_path for pattern in excludes):
                continue

            directory_config: dict = {}
            for part in Path(file_path).parts:
                if part in METADATA_DIRECTORY_CONFIG:
                    directory_config = METADATA_DIRECTORY_CONFIG[part]
                    break

            ignore_patterns = directory_config.get("ignore_patterns", [])
            if ignore_patterns and any(pattern in file_path for pattern in ignore_patterns):
                continue

            yield file_path
