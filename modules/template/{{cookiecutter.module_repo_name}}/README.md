# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

A **Holoscan Module** — a self-contained, redistributable library that extends
[Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) with reusable operators under the
`holoscan.{{ cookiecutter.module_slug }}` namespace.
{%- set op_class = cookiecutter.operator_slug.split('_')|map('capitalize')|join('') %}
{%- set mod_kebab = cookiecutter.module_slug.replace('_', '-') %}

---

## Quick Start

```bash
# 1. Run the Python demo application
./holohub run {{ cookiecutter.module_slug }}_pipeline --language python
{% if cookiecutter.language == 'cpp' %}
# 2. Run the C++ demo application
./holohub run {{ cookiecutter.module_slug }}_pipeline
{% endif %}
```

---

## Operators

| Operator | Implementation | Ports | Parameters |
|---|---|---|---|
| `{{ op_class }}` | {% if cookiecutter.language == 'cpp' %}C++ + pybind11{% else %}Pure Python{% endif %} | TODO | TODO |

**Namespace**
{% if cookiecutter.language == 'cpp' %}

- C++: `holoscan::{{ cookiecutter.module_slug }}`
{% endif %}
- Python: `holoscan.{{ cookiecutter.module_slug }}`

---

## Usage

### Python

```python
from holoscan.core import Application
from holoscan.{{ cookiecutter.module_slug }} import {{ op_class }}


class MyApp(Application):
    def compose(self):
        op = {{ op_class }}(self, name="{{ cookiecutter.operator_slug }}")
        # TODO: connect operators and add_flow calls


MyApp().run()
```

{% if cookiecutter.language == 'cpp' %}

### C++

```cpp
#include <holoscan/holoscan.hpp>
#include <{{ cookiecutter.operator_slug }}/{{ cookiecutter.operator_slug }}.hpp>

class MyApp : public holoscan::Application {
 public:
  void compose() override {
    auto op = make_operator<holoscan::{{ cookiecutter.module_slug }}::{{ op_class }}>("{{ cookiecutter.operator_slug }}");
    // TODO: connect operators
  }
};

int main() { holoscan::make_application<MyApp>()->run(); }
```

{% endif %}

---

## Building from Source (without HoloHub CLI)

| Requirement | Version |
|---|---|
| Holoscan SDK | ≥ {{ cookiecutter.holoscan_version }} |
| CUDA Toolkit | 13.x (matches the Holoscan SDK CUDA pin; the dev `Dockerfile` uses `cuda13-dgpu`) |
| CMake | ≥ 3.24 |
{%- if cookiecutter.language == 'cpp' %}
| C++ compiler | C++17 (GCC 11+) |
| pybind11 | ≥ 2.11 |
{%- endif %}
| Python | 3.10–3.13 |

```bash
cmake -S . -B build -DBUILD_ALL=ON -D{{ cookiecutter.module_slug | upper }}_BUILD_TESTING=ON
cmake --build build -j$(nproc)
```

---

## Testing

```bash
./holohub test
```

Or, without the HoloHub CLI:

{% if cookiecutter.language == 'cpp' %}

```bash
# C++ (GTest via CTest)
ctest --test-dir build --output-on-failure -L unit
```

{% endif %}

```bash
# Python (pytest)
PYTHONPATH=build/python${PYTHONPATH:+:$PYTHONPATH} {{ cookiecutter.module_slug | upper }}_BUILD_DIR=build pytest tests/python/ -v
```

`PYTHONPATH` is **prepended** (with `${PYTHONPATH:+:$PYTHONPATH}`) so that an
ambient holoscan SDK install stays visible. A bare `PYTHONPATH=build/python`
would replace the variable and hide the SDK from pytest. A bare
`PYTHONPATH=build/python:$PYTHONPATH` looks safe but, when `$PYTHONPATH` is
unset (typical on a fresh shell or CI runner), it expands to a trailing colon
that Python reads as an empty path entry — equivalent to `.`, which silently
adds the test CWD to `sys.path` and lets a local file shadow installed
packages.

The pytest suite currently covers importability and build-smoke only; full
live-pipeline coverage is a TODO and may require real hardware. CTest is
configured with `SKIP_RETURN_CODE 5` on the pytest entry — if a CTest run
reports "Skipped" unexpectedly, the most common cause is that the holoscan
SDK is not importable in the test environment (pytest collects zero items
and exits 5).

---

## License

{{ cookiecutter._license }} — see [LICENSE](LICENSE).
