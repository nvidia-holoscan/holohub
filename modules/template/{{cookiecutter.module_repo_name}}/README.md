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
PYTHONPATH=build/python {{ cookiecutter.module_slug | upper }}_BUILD_DIR=build pytest tests/python/ -v
```

---

## License

{{ cookiecutter._license }} — see [LICENSE](LICENSE).
