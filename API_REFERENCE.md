# HoloHub API Reference

This document is a quick reference for HoloHub metadata, build system, directory layout, path placeholders, naming conventions, and testing. For contribution workflow, checklists, and process, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Table of Contents

- [Metadata Reference](#metadata-reference)
- [Path Placeholders](#path-placeholders)
- [Build System Reference](#build-system-reference)
- [Directory Structure Reference](#directory-structure-reference)
- [Naming Conventions](#naming-conventions)
- [CLI Command Reference](#cli-command-reference)
- [Testing Reference](#testing-reference)
- [See Also](#see-also)

---

## Metadata Reference

Every application and operator must have a `metadata.json` that describes features and dependencies. Contribution-specific schemas define the exact fields.

### Schema Locations

| Type | Schema path |
|------|-------------|
| Workflows | [workflows/metadata.schema.json](workflows/metadata.schema.json) |
| Applications | [applications/metadata.schema.json](applications/metadata.schema.json) |
| GXF Extensions | [gxf_extensions/metadata.schema.json](gxf_extensions/metadata.schema.json) |
| Operators | [operators/metadata.schema.json](operators/metadata.schema.json) |
| Tutorials | [tutorials/metadata.schema.json](tutorials/metadata.schema.json) |

### metadata.json Structure (Overview)

Root key is `application` or `operator` depending on contribution type. Main fields:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Explicit name of the application or operator |
| `authors` | array | Objects with `name` and `affiliation` |
| `language` | string | `C++`, `Python`, or `GXF`; use one directory per language if multiple |
| `version` | string | Semantic version: `major.minor.patch` |
| `changelog` | object | Keys are versions (for example `"X.X"`), values are short change descriptions |
| `holoscan_sdk` | object | `minimum_required_version`, `tested_versions` (array) |
| `platforms` | array | for example `["x86_64", "aarch64"]` |
| `tags` | array | Free-form tags (for example `["Endoscopy", "Video Encoding"]`) |
| `ranking` | number | Self-assessment level 0–5; see [Ranking levels](#ranking-levels-for-metadatajson) |
| `dependencies` | object | for example `operators: [{ name, version }]` |
| `run` | object | (Applications) `command` string and `workdir`; see [Path placeholders](#path-placeholders) |

**Example snippet:**

```json
"application": {
    "name": "my_application",
    "authors": [{ "name": "Your Name", "affiliation": "Your Organization" }],
    "language": "C++",
    "version": "1.0.0",
    "holoscan_sdk": {
        "minimum_required_version": "0.6.0",
        "tested_versions": ["0.6.0"]
    },
    "platforms": ["x86_64", "aarch64"],
    "tags": ["Video"],
    "ranking": 2,
    "run": {
        "command": "./myapplication --data <holohub_data_dir>/mydata",
        "workdir": "holohub_app_bin"
    }
}
```

### Ranking Levels for metadata.json

| Level | Description | Requirements |
|-------|-------------|--------------|
| **0** | Production-ready, SDK-level quality | Widespread community dependence; above 90% code coverage; nightly testing; all Level 1 requirements |
| **1** | Very high-quality code | Holoscan SDK coding standards; builds on all platforms within 1 month of releases; 75% code coverage; CI testing; all Level 2 requirements |
| **2** | Quality code | Compiles on community platforms; may have external dependencies; tests pass on supported platforms; all Level 3 requirements |
| **3** | Features under development | Builds on specific platforms; some tests passing; all Level 4 requirements |
| **4** | Code of unknown quality | Builds on specific platforms; minimal test coverage |
| **5** | Deprecated | Limited utility; may have known bugs |

---

## Path Placeholders

In `metadata.json`, the `workdir` for the run command can be one of these. They are resolved at build/run time:

| Placeholder | Meaning |
|-------------|---------|
| `holohub_app_bin` | Directory containing the built application binary (for example `<holohub_root>/build/myapp/applications/myapp/cpp`) |
| `holohub_app_source` | Application source directory (for example `<holohub_root>/applications/myapp/cpp/`) |
| `holohub_bin` | Root build directory (for example `<holohub_root>/build/`) |
| `holohub_data_dir` | Data directory (for example `<holohub_root>/data/`) |

Use `<holohub_data_dir>` in the run `command` for data paths (for example `--data <holohub_data_dir>/mydata`).

---

## Build System Reference

HoloHub uses CMake. Register your contribution in the appropriate top-level `CMakeLists.txt` and, for packages, in package config.

### Operators

**File to edit:** `./operators/CMakeLists.txt`

```cmake
add_holohub_operator(my_operator DEPENDS EXTENSIONS my_extension)
```

- `my_operator`: operator target name (matches directory under `operators/`).
- `DEPENDS EXTENSIONS` (optional): list of GXF extension targets this operator wraps; build system will build them first.

### GXF Extensions

**File to edit:** `./gxf_extensions/CMakeLists.txt`

```cmake
add_holohub_extension(my_extension)
```

### Applications

**File to edit:** `./applications/CMakeLists.txt`

```cmake
add_holohub_application(my_application DEPENDS OPERATORS my_operator1 my_operator2)
```

- `DEPENDS OPERATORS` (optional): operator targets required by this application.

### Workflows

**File to edit:** `./workflows/CMakeLists.txt`

```cmake
add_holohub_application(my_workflow DEPENDS OPERATORS my_operator1 my_operator2)
```

Same pattern as applications; use `add_holohub_application` with optional `DEPENDS OPERATORS`.

### Packages (DEB)

**Package CMake config** (in `./pkg/my_package/CMakeLists.txt`):

```cmake
holohub_configure_deb(
  NAME "my-package-dev"
  COMPONENTS "my-headers" "my-libs"   # optional
  DESCRIPTION "My project description"
  VERSION "1.0.0"
  VENDOR "Your Organization"
  CONTACT "Your Name <your.email@example.com>"
  DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11)"
  SECTION "devel"      # optional
  PRIORITY "optional"  # optional
)
```

**Package registration** (in `./pkg/CMakeLists.txt`):

```cmake
add_holohub_package(my_packager
                    APPLICATIONS my_app1
                    OPERATORS my_op1 my_op2)
```

CMake targets must have `install` rules; use `COMPONENT` to control packaging granularity.

---

## Directory Structure Reference

Required and optional files per contribution type.

### Operators

```text
holohub/operators/your_operator_name/
├── metadata.json                   # Required; operator schema
├── README.md                       # Required
├── your_operator_name.py | .cpp | .hpp
├── test_your_operator_name.py      # Required for Python operators
└── CMakeLists.txt                  # If needed for C++
```

### Applications

```text
holohub/applications/your_app_name/
├── metadata.json                   # Required; application schema
├── README.md                       # Required
├── your_app_name.py | .cpp
└── CMakeLists.txt
```

### Workflows

```text
holohub/workflows/your_workflow_name/
├── metadata.json                   # Required; workflow schema
├── README.md                       # Required
├── your_workflow_name.py | .cpp
└── CMakeLists.txt
```

### GXF Extensions

```text
holohub/gxf_extensions/your_extension_name/
├── metadata.json                   # Required; extension schema
├── README.md                       # Required
├── your_extension.cpp
├── your_extension.hpp
└── CMakeLists.txt                  # Required
```

### Tutorials

```text
holohub/tutorials/your_tutorial_name/
├── README.md                       # Required
├── metadata.json                   # Optional; tutorial schema
├── tutorial_code.py | .cpp
└── assets/                         # Optional
```

### Packages

```text
holohub/pkg/your_package_name/
├── CMakeLists.txt                  # Required
└── README.md                       # Optional
```

---

## Naming Conventions

Example for an operator named "Adaptive Thresholding":

| Component | Convention | Example |
|-----------|------------|---------|
| Class name | TitleCase + `Op` suffix | `AdaptiveThresholdingOp` |
| metadata.json `name` | Same as class name | `AdaptiveThresholdingOp` |
| Directory | snake_case | `adaptive_thresholding` |
| Filename | Same as directory + extension | `adaptive_thresholding.py` |
| README title | Title Case + "Operator" | "Adaptive Thresholding Operator" |
| Unit test file | `test_` + directory name | `test_adaptive_thresholding.py` |

---

## CLI Command Reference

Commands used in the contribution workflow. For full options and behavior, see [utilities/cli/CLI_REFERENCE.md](utilities/cli/CLI_REFERENCE.md).

| Command | Purpose |
|---------|---------|
| `./holohub create <application_name>` | Generate application scaffolding (directory structure and initial files). |
| `./holohub lint` | Run lint checks (optionally `./holohub lint path/to/code`). |
| `./holohub lint --install-dependencies` | Install lint tooling (may require `sudo`). |
| `./holohub lint --fix` | Auto-fix common lint issues. |
| `./holohub test <project>` | Run tests for a project (CTest). |

---

## Testing Reference

### Integration and CTest

- Each operator should have at least one application under [applications/](applications/) that demonstrates it.
- Applications add a testing section in their `CMakeLists.txt`. HoloHub uses [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html).

### Python Operator Unit Tests

- **Framework:** `pytest`
- **File:** `test_<operator_name>.py` in the same directory as the operator.
- **Fixtures:** Reuse common fixtures from `conftest.py`.

**Test categories:**

1. Initialization: operator creation and properties.
2. Port setup: input/output ports configured correctly.
3. Error handling: invalid arguments with `pytest.raises`.
4. Compute logic: main functionality with various inputs.
5. Edge cases: boundary conditions and errors.

**Run commands:**

```bash
# From repository root
pytest operators/<your_operator_dir>/

# With coverage
pytest --cov=operators/<your_operator_dir>/ operators/<your_operator_dir>/

# Verbose
pytest -v operators/<your_operator_dir>/
```

**Example test pattern:**

```python
import pytest
from .my_operator import MyOperatorOp
from holoscan.core import Operator, _Operator as BaseOperator

def test_my_operator_init(fragment):
    op = MyOperatorOp(fragment=fragment, name="myoperator_op", tensor_name="image")
    assert isinstance(op, BaseOperator)
    assert op.operator_type == Operator.OperatorType.NATIVE

def test_my_operator_invalid_param(fragment):
    with pytest.raises(ValueError):
        MyOperatorOp(fragment=fragment, tensor_name="image", invalid_param=-1)
```

Reference implementations: for example `operators/deidentification/pixelator/test_pixelator.py` and `conftest.py` for fixtures.

---

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution workflow, checklists, and guidelines
- [utilities/cli/CLI_REFERENCE.md](utilities/cli/CLI_REFERENCE.md) — Full HoloHub CLI reference
- [README.md](README.md) — Project overview and glossary
- [doc/developer.md](doc/developer.md) — Advanced developer guide
