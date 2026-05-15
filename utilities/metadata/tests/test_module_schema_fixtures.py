# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Schema-regression suite for module.schema.json (v2). Validates the
# hand-curated fixtures under tests/fixtures/{valid,invalid}/ — each
# valid file must pass the module schema, and each invalid file must
# fail with jsonschema.exceptions.ValidationError. The fixtures pin
# down the contract that the external_resolver + CLI rely on:
# dependency_source requires git_url + ref, module_dependency forbids
# extra fields, dependencies is an array, etc.

from __future__ import annotations

import json
import unittest
from pathlib import Path

import jsonschema
from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.jsonschema import DRAFT202012

THIS_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = THIS_DIR.parent  # utilities/metadata/
FIXTURE_DIR = THIS_DIR / "fixtures"


def _make_module_validator() -> Draft202012Validator:
    """Build a Draft 2020-12 validator for module.schema.json with project.schema.json
    registered as a referenceable resource (the module schema $refs
    urn:holohub:project:v1#/$defs/...)."""
    base_schema = json.loads((SCHEMA_DIR / "project.schema.json").read_text())
    module_schema = json.loads((SCHEMA_DIR / "module.schema.json").read_text())
    registry = Registry().with_resource(
        base_schema["$id"], DRAFT202012.create_resource(base_schema)
    )
    return Draft202012Validator(module_schema, registry=registry)


class TestModuleSchemaFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = _make_module_validator()

    def test_valid_fixtures_pass(self) -> None:
        valid_dir = FIXTURE_DIR / "valid"
        fixtures = sorted(valid_dir.glob("*.json"))
        self.assertGreater(len(fixtures), 0, f"no valid fixtures found under {valid_dir}")
        for fx in fixtures:
            with self.subTest(fixture=fx.name):
                data = json.loads(fx.read_text())
                # Will raise on failure; subTest reports the offender.
                self.validator.validate(data)

    def test_invalid_fixtures_fail(self) -> None:
        invalid_dir = FIXTURE_DIR / "invalid"
        fixtures = sorted(invalid_dir.glob("*.json"))
        self.assertGreater(len(fixtures), 0, f"no invalid fixtures found under {invalid_dir}")
        for fx in fixtures:
            with self.subTest(fixture=fx.name):
                data = json.loads(fx.read_text())
                with self.assertRaises(
                    jsonschema.exceptions.ValidationError,
                    msg=f"{fx.name} should be invalid but passed schema",
                ):
                    self.validator.validate(data)


if __name__ == "__main__":
    unittest.main()
