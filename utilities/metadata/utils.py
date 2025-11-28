# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Iterable


def normalize_language(language: str | None, *, strict: bool = False) -> str:
    """
    Normalize language names, optionally enforcing known HoloHub languages.

    Args:
        language: The language name to normalize (e.g., "cpp", "python", "c++", "py").
        strict: If True, raises ValueError for languages other than "cpp" and "python".

    Returns:
        Normalized language name ("cpp", "python", or the lowercased input if strict=False).
        Returns empty string if language is None or not a string.

    Raises:
        ValueError: If strict=True and language is not a recognized HoloHub language.
    """
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
    """
    Return a list of normalized language tags from a single value or sequence.

    Args:
        language: A language string, list/iterable of languages, or None.
        strict: If True, raises ValueError for unrecognized languages (passed to normalize_language).

    Returns:
        List of normalized language strings. Returns [""] if no valid languages found.
    """
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
