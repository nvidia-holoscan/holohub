# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validates the three static metadata.json.template files that ship with the
# holohub application, operator, and tutorial scaffolding tools. These files
# are excluded from the corpus validator (metadata_validator.py) by design, so
# this suite provides the only automated schema-correctness check for them.
# It specifically guards the "$schema" header addition and the platform enum
# fix (x86_64/aarch64) that were introduced alongside the Draft 2020-12
# migration.

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.jsonschema import DRAFT202012

THIS_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = THIS_DIR.parent  # utilities/metadata/
REPO_ROOT = SCHEMA_DIR.parents[1]  # holohub/

# (template_path_relative_to_repo_root, schema_name)
_TEMPLATE_CASES = [
    ("applications/template/metadata.json.template", "application"),
    ("operators/template/metadata.json.template", "operator"),
    ("tutorials/template/metadata.json.template", "tutorial"),
]


def _make_validator(schema_name: str) -> Draft202012Validator:
    base_schema = json.loads((SCHEMA_DIR / "project.schema.json").read_text())
    entity_schema = json.loads((SCHEMA_DIR / f"{schema_name}.schema.json").read_text())
    registry = Registry().with_resource(
        base_schema["$id"], DRAFT202012.create_resource(base_schema)
    )
    return Draft202012Validator(entity_schema, registry=registry)


class TestStaticTemplateFiles(unittest.TestCase):
    def test_static_templates_are_schema_valid(self) -> None:
        for rel_path, schema_name in _TEMPLATE_CASES:
            with self.subTest(template=rel_path):
                full_path = REPO_ROOT / rel_path
                self.assertTrue(full_path.exists(), f"template file not found: {full_path}")
                data = json.loads(full_path.read_text())
                validator = _make_validator(schema_name)
                validator.validate(data)
                schema_id = json.loads((SCHEMA_DIR / f"{schema_name}.schema.json").read_text())[
                    "$id"
                ]
                self.assertEqual(
                    data.get("$schema"),
                    schema_id,
                    f"$schema in {rel_path} does not match schema $id {schema_id!r}",
                )


if __name__ == "__main__":
    unittest.main()
