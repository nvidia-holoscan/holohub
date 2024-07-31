# Template for Holoviz Examples

Holoviz examples are generated from template files using Cookiecutter. The generated code is checked in, to re-generate the code after changing the template, execute the `generate_projects.sh` script.

## Adding a new example

Add a new value to the `examples` array in `cookiecutter-holoviz/cookiecutter.json`, e.g. `"new_example"`.
Add a new line `generate "new_example_dir" "new_example" "New Example Window Title"` to `generate_projects.sh`.
Modify the source code file `cookiecutter-holoviz/{{cookiecutter.project_slug}}/{{cookiecutter.project_slug}}.cpp` and add new
code covered by the

```
{%- if cookiecutter.example == "new_example" %}
    // some new code
{%- endif %}
```

Add a screenshot of the app as `new_example_dir.png` to the `cookiecutter-holoviz` directory. This will be shown in the readme.

Add `add_holohub_application(holoviz_new_example)` to the `CMakeLists.txt` file in `applications\holoviz`.

Execute the `generate_projects.sh` script to generate the new example.