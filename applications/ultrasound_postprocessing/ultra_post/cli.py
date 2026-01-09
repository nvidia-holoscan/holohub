# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ultra_post.core.pipeline import pipeline_from_yaml
from ultra_post.filters.registry import DEFAULT_PARAMS, FILTERS


def _load_filters() -> dict[str, Any] | None:
    return dict(FILTERS)


def _cmd_list_filters(_args: argparse.Namespace) -> int:
    filters = _load_filters()
    exit_code = 1
    if filters:
        for name in sorted(filters):
            defaults = DEFAULT_PARAMS.get(name, {})
            sig = defaults if defaults else "(no defaults)"
            print(f"{name}: {sig}")
        exit_code = 0
    else:
        print("No filters found.")
    return exit_code


def _cmd_validate_preset(args: argparse.Namespace) -> int:
    path = Path(args.path)
    exit_code = 1
    if not path.exists():
        print(f"Preset not found: {path}", file=sys.stderr)
    else:
        filters = _load_filters()
        if filters:
            try:
                text = path.read_text(encoding="utf-8")
                pipeline_from_yaml(text, filters=filters)
                print(f"Preset '{path}' is valid.")
                exit_code = 0
            except Exception as exc:  # pragma: no cover - CLI validation path
                print(f"Preset validation failed: {exc}", file=sys.stderr)
                exit_code = 2
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ultrasound post-processing CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-filters", help="List discovered operators").set_defaults(func=_cmd_list_filters)

    validate = sub.add_parser("validate-preset", help="Validate a pipeline preset YAML")
    validate.add_argument("path", type=str, help="Path to preset YAML")
    validate.set_defaults(func=_cmd_validate_preset)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
