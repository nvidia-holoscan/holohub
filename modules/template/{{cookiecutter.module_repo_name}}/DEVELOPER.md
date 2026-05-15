# Developer Guide — {{ cookiecutter.project_name }}

{%- set op_class = cookiecutter.operator_slug.split('_')|map('capitalize')|join('') %}

This guide covers the layout, build system, and day-to-day workflow for developing and
distributing this Holoscan Module.

---

## Module layout

```
{{ cookiecutter.module_repo_name }}/
├── holohub                         # CLI wrapper (delegates to HoloHub CLI)
├── Dockerfile                      # Dev container image
├── CMakeLists.txt                  # Root CMake — orchestrates operators/applications/tests
├── pyproject.toml                  # Python packaging metadata (scikit-build-core)
├── metadata.json                   # Module-level metadata (schema: holoscan/module/v2)
├── operators/
│   └── {{ cookiecutter.operator_slug }}/
│       ├── {{ cookiecutter.operator_slug }}.{% if cookiecutter.language == 'cpp' %}cpp / .hpp{% else %}py{% endif %}  # Operator implementation
│       └── metadata.json           # Operator-level metadata
├── applications/
│   └── {{ cookiecutter.module_slug }}_pipeline/
│       └── {{ cookiecutter.module_slug }}_pipeline.{% if cookiecutter.language == 'cpp' %}cpp{% else %}py{% endif %}   # Example pipeline
├── python/holoscan/{{ cookiecutter.module_slug }}/
│   └── __init__.py                 # Re-exports operators for `from holoscan.{{ cookiecutter.module_slug }} import ...`
└── tests/
    ├── cpp/                        # GTest suite (C++ modules only)
    └── python/                     # pytest suite
```

---

## `holohub` wrapper commands

The `holohub` script at the module root wraps the HoloHub CLI with module-specific defaults.
On first run it downloads the CLI tools via sparse-checkout into `.holohub/` (internet required;
subsequent runs use the cached copy).

| Command | What it does |
|---|---|
| `./holohub run-container` | Build and start the dev container |
| `./holohub build {{ cookiecutter.module_slug }}_pipeline` | CMake configure + build inside the container |
| `./holohub run {{ cookiecutter.module_slug }}_pipeline` | Run the example pipeline |
| `./holohub test` | Run CTest (C++ unit tests) and pytest |
| `./holohub install --dev` | Install a `.pth` hook so `import holoscan.{{ cookiecutter.module_slug }}` works in any shell |

Set `CLI_FORCE_UPDATE=1` to re-download the CLI tools (e.g. after updating `CLI_PINNED_COMMIT`).

---

## Building without the wrapper

```bash
cmake -S . -B build -DBUILD_ALL=ON -D{{ cookiecutter.module_slug | upper }}_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
```

{% if cookiecutter.language == 'cpp' %}
Run C++ tests:

```bash
ctest --test-dir build --output-on-failure -L unit
```

{% endif %}
Run Python tests:

```bash
{{ cookiecutter.module_slug | upper }}_BUILD_DIR=build \
PYTHONPATH=build/python \
pytest tests/python/ -v
```

---

## `pyproject.toml`

`pyproject.toml` configures [scikit-build-core](https://scikit-build-core.readthedocs.io/) for
wheel packaging. Key fields to update before publishing:

| Field | Purpose |
|---|---|
| `[project].name` | PyPI package name — should match `metadata.json:module.binary_packages.pypi` |
| `[project].version` | Sync with `metadata.json:module.version` |
| `[project].description` | Short description shown on PyPI |
| `[project].authors` | Your name / organisation |
| `[tool.scikit-build].cmake.args` | Extra CMake flags passed during `pip install` |

Build a wheel:

```bash
pip install build
python -m build --wheel
```

---

## Naming conventions

| Context | Convention | Example |
|---|---|---|
| Python import / C++ namespace | `snake_case` | `holoscan.{{ cookiecutter.module_slug }}` |
| Repository folder | `holoscan-<slug>` (kebab) | `{{ cookiecutter.module_repo_name }}` |
| Debian package | `holoscan-<slug>` (kebab) | `holoscan-{{ cookiecutter.module_slug.replace('_', '-') }}` |
| PyPI package | `holoscan-<slug>` (kebab) | `holoscan-{{ cookiecutter.module_slug.replace('_', '-') }}` |
| CMake option prefix | `UPPER_SNAKE` | `{{ cookiecutter.module_slug | upper }}_BUILD_TESTING` |

---

## Further reading

- [HoloHub documentation](https://github.com/nvidia-holoscan/holohub)
- [Holoscan SDK documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/)
- [Holoscan Module ecosystem](https://nvidia-holoscan.github.io/)
