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

from pathlib import Path

import pytest
from ultra_post.core.pipeline import pipeline_from_yaml
from ultra_post.filters.registry import FILTERS

PRESETS_DIR = Path(__file__).parent.parent / "presets"


def get_preset_files() -> list[Path]:
    return list(PRESETS_DIR.glob("*.y*ml"))


@pytest.fixture
def filters():
    return FILTERS


class TestPresets:
    def test_presets_exist(self) -> None:
        preset_files = get_preset_files()
        assert preset_files, f"No preset files found in {PRESETS_DIR}"

    @pytest.mark.parametrize("preset_file", get_preset_files(), ids=lambda p: p.name)
    def test_preset_loads(self, preset_file: Path, filters) -> None:
        text = preset_file.read_text(encoding="utf-8")
        pipeline = pipeline_from_yaml(text, filters=filters)
        assert pipeline, f"Pipeline is empty for {preset_file.name}"
