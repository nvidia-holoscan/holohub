# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for ``$schema`` label checks and the deprecated ``urn:holohub:`` alias."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from utilities.metadata import metadata_validator


class SchemaLabelTest(unittest.TestCase):
    def validate(self, label):
        return metadata_validator.validate_json({"$schema": label, "package": {}}, "pkg")

    def test_canonical_label_is_accepted_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ok, msg = self.validate("urn:holoscan:package:v1")
        self.assertTrue(ok, msg)

    def test_missing_label_is_accepted(self):
        ok, msg = metadata_validator.validate_json({"package": {}}, "pkg")
        self.assertTrue(ok, msg)

    def test_legacy_label_is_accepted_with_warning(self):
        with self.assertWarnsRegex(FutureWarning, "2027"):
            ok, msg = self.validate("urn:holohub:package:v1")
        self.assertTrue(ok, msg)

    def test_mismatched_labels_are_rejected(self):
        for label in (
            "urn:holoscan:module:v2",
            "urn:holoscan:package:v2",
            "holoscan/package/v1",
            "urn:example:package:v1",
            1,
        ):
            with self.subTest(label=label):
                ok, msg = self.validate(label)
                self.assertFalse(ok)
                self.assertIn('"$schema"', msg)

    def test_legacy_ref_alias_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = Path(tmp) / "legacy.schema.json"
            schema.write_text(
                json.dumps(
                    {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "$id": "urn:holoscan:legacy:v1",
                        "type": "object",
                        "properties": {"tags": {"$ref": "urn:holohub:project:v1#/$defs/tags"}},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(metadata_validator, "get_schema_path", return_value=schema):
                ok, msg = metadata_validator.validate_json({"tags": ["a"]}, "legacy")
                self.assertTrue(ok, msg)
                ok, _ = metadata_validator.validate_json({"tags": [1]}, "legacy")
                self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
