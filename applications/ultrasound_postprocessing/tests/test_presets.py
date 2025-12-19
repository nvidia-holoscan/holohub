# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from pathlib import Path

from ultra_post.core.pipeline import pipeline_from_yaml
from ultra_post.filters.registry import FILTERS


class PresetTests(unittest.TestCase):
    def test_all_presets_load(self) -> None:
        presets_dir = Path("presets")
        filters = FILTERS
        for preset in presets_dir.glob("*.y*ml"):
            text = preset.read_text(encoding="utf-8")
            pipeline_from_yaml(text, filters=filters)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
