# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is HoloHub

HoloHub is a community repository for NVIDIA Holoscan SDK — a platform for real-time AI sensor processing. It contains reference applications, reusable operators, GXF extensions, tutorials, and benchmarks. Holoscan SDK applications are directed acyclic graphs of **Operators** (processing nodes) connected by ports.

## Key Commands

All operations use the `./holohub` CLI script (a thin wrapper around `utilities/cli/holohub.py`):

```bash
./holohub list                              # List all available components
./holohub build <app|operator|package>      # Build a specific component
./holohub run <application_name>            # Build container + run application
./holohub run <app> --language=python       # Run specific language implementation
./holohub run <app> --run-args='...'        # Pass extra args to application
./holohub build-container [project_name]    # Build the dev container
./holohub run-container [project_name]      # Launch the dev container interactively
./holohub lint                              # Lint entire repository
./holohub lint path/to/code                 # Lint specific path
./holohub lint --fix                        # Auto-fix lint issues
./holohub lint --install-dependencies       # Install lint tools
./holohub test <project>                    # Run CTest for a project
./holohub setup                             # Install native build dependencies
./holohub create <app_name>                 # Scaffold new application
./holohub clear-cache                       # Clear build/data/install cache
```

### Running Python Unit Tests

Tests live alongside operators in `operators/<name>/test_<name>.py`:

```bash
# From repo root
pytest operators/<operator_dir>/
pytest -v operators/<operator_dir>/
pytest --cov=operators/<operator_dir>/ operators/<operator_dir>/
```

### Manual CMake Build

```bash
cmake -S . -B build \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DHOLOHUB_DATA_DIR=$(pwd)/data \
  -DAPP_<application_name>=1
cmake --build build
```

## Repository Structure

```
holohub/
├── applications/          # Example applications (each with metadata.json + README.md + code)
├── operators/             # Reusable Holoscan operators (Python/C++)
├── gxf_extensions/        # GXF (Graph Execution Framework) extensions (C++)
├── workflows/             # End-to-end "sensor-to-insight" pipelines
├── tutorials/             # Extended walkthroughs
├── benchmarks/            # Performance benchmarking tools
├── pkg/                   # Debian package configurations
├── cmake/                 # CMake modules and HoloHubConfigHelpers.cmake
├── utilities/
│   ├── cli/               # holohub CLI (holohub.py, container.py)
│   └── metadata/          # metadata.json parsing utilities
├── conftest.py            # Root pytest fixtures shared across operator tests
├── pyproject.toml         # Python tooling config (black, ruff, isort, pytest)
└── holohub                # Entry point shell script → utilities/cli/holohub.py
```

## Architecture: How Applications and Operators Are Built

The CMake build system uses helper functions from `cmake/HoloHubConfigHelpers.cmake`:

- `add_holohub_application(name DEPENDS OPERATORS op1 op2)` — in `applications/CMakeLists.txt`
- `add_holohub_operator(name DEPENDS EXTENSIONS ext1)` — in `operators/CMakeLists.txt`
- `add_holohub_extension(name)` — in `gxf_extensions/CMakeLists.txt`
- `add_holohub_package(name APPLICATIONS app1 OPERATORS op1)` — in `pkg/CMakeLists.txt`

Each component is controlled by a CMake option (`APP_<name>`, `OP_<name>`, `EXT_<name>`, `PKG_<name>`). Setting `BUILD_ALL=ON` enables all components.

Build artifacts go to `./build/<component_name>/` (isolated per component). Data is downloaded to `./data/` at build time.

## Every Component Requires metadata.json

Every application, operator, workflow, tutorial, and GXF extension needs a `metadata.json` file. Schemas:
- `applications/metadata.schema.json`
- `operators/metadata.schema.json`
- `workflows/metadata.schema.json`
- `gxf_extensions/metadata.schema.json`
- `tutorials/metadata.schema.json`

The **first tag** in the `tags` array is the category. Valid categories are documented in `.github/copilot-instructions.md`. The `run.workdir` field must be one of: `holohub_app_bin`, `holohub_app_source`, or `holohub_bin`.

## Naming Conventions

| Component | Convention | Example |
|-----------|-----------|---------|
| Operator class | TitleCase + "Op" | `AdaptiveThresholdingOp` |
| Directory | snake_case | `adaptive_thresholding` |
| Filename | matches directory | `adaptive_thresholding.py` |
| Unit test | `test_` + dirname | `test_adaptive_thresholding.py` |

## Python Operator Tests

The root `conftest.py` provides shared fixtures: `app`, `fragment`, `config_file`, `mock_image`, `op_input_factory`, `op_output`, `execution_context`. Tests for Python operators use these fixtures and cover:
1. Initialization (verify class type, operator type, name in repr)
2. Port setup (check `spec.inputs` and `spec.outputs`)
3. Error handling with `pytest.raises`
4. Compute logic with parametrized inputs

## Code Style

- Python: `black` (line length 100), `ruff`, `isort` (profile: black) — configured in `pyproject.toml`
- Commits must be signed-off: `git commit -s -m "..."` (required by DCO)
- License: Apache 2.0 — all source files need the SPDX license header

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOLOHUB_BUILD_PARENT_DIR` | `<repo>/build` | Override build directory |
| `HOLOHUB_DATA_DIR` | `<repo>/data` | Override data directory |
| `HOLOHUB_DEFAULT_HSDK_DIR` | `/opt/nvidia/holoscan` | Holoscan SDK install path |
