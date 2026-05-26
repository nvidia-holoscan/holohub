# Tutorial: Create a Holoscan Module

In this tutorial you will learn how to create a new **Holoscan Module** extending the Holoscan SDK.
We will walk through project setup from a blank directory to a published community module.

## 1. Holoscan Modules in Context

### What is NVIDIA Holoscan?

**Holoscan SDK** is NVIDIA's platform for building streaming, low-latency AI pipelines
targeting medical devices, robotics, and industrial edge systems. Applications in the
ecosystem are composed of *operators* — discrete processing units connected into a
directed acyclic graph (DAG) by the SDK's pipeline executor.

### What is a Holoscan Module?

A **Holoscan Module** is a distributable library of one or more related operators, subgraphs, or
other components to extend the Holoscan SDK application programming interface (API). A module
may be packaged and published so that any Holoscan application can consume it as a binary
dependency. It is the standard mechanism for sharing reusable Holoscan processing blocks
beyond a single application or repository. Consumers get a working, namespaced import in one command:

```bash
pip install holoscan-my-sensor          # install
from holoscan.my_sensor import MySensorOp  # use
```

without cloning your source or understanding your build system.

Each Holoscan Module typically covers a set of components that are related by some domain or
hardware context. For example:

- a `holoscan-<my-camera>` module may distribute a set of operators
with logic specific to a certain class of cameras;
- a `holoscan-image-reconstruction` module may distribute a set of resources and operators
targeting applications in the domain of image reconstruction.

There are two flavors of modules:

- **External module** — a standalone git repository with its own operators,
  applications, tests, packaging, and CI. Consumers fetch it at build time. *This is the
  primary path covered in this tutorial.*
- **HoloHub-hosted module** — a thin descriptor inside the HoloHub monorepo whose operator
  sources live in `operators/<name>/`. See `modules/holoscan-gstreamer/` for the
  canonical reference, and Section 4 below for the steps.

In this tutorial we will set up a fictional `holoscan-my-sensor` external module.

### Who creates a Holoscan Module?

A Holoscan Module is the right artifact to build when:

- **You are a hardware vendor** who wants developers to integrate your sensor, camera, or
  accelerator into Holoscan pipelines without needing your source code.
- **You are a domain expert** (medical imaging, robotics, industrial vision) who has built
  signal-processing or AI-inference operators that are broadly reusable and wants to share
  them across the ecosystem.
- **You have operators in a HoloHub sample project** that have matured and are now worth
  publishing as a first-class, versioned package independent of the full HoloHub
  monorepo.
- **You need reproducible, release-quality binaries** — a Python wheel and/or Debian
  package pinned to a specific Holoscan SDK version — for integration into a regulated or
  production pipeline.

In each case the goal is the same: give consumers a `pip install` or `apt install`
experience backed by operators developed and tested against a known Holoscan SDK version,
with discovery through the Holoscan ecosystem site rather than word of mouth.

### Why build a Holoscan Module instead of starting from a blank project?

1. **Discovery.** Curated Holoscan Modules may be listed alongside SDK-supported and community work
   on the NVIDIA HoloHub website. Prospective users and their AI agents find your library
   by searching the Holoscan ecosystem instead of guessing repo names. A blank template
   puts the entire burden of discovery on you.
2. **Zero-to-scaffold in one command.** `./holohub create … --template modules/template`
   produces a working CMake + scikit-build-core build, a Dockerfile pinned to a known
   Holoscan SDK image, pybind11 bindings wired up, a passing test layout, and CI
   workflow stubs. Cookiecutter prompts keep the slug, namespace, and package name
   consistent so the result imports cleanly as `holoscan.<your_slug>`.
3. **CLI built for the Holoscan lifecycle.** `./holohub build`, `test`, `run`,
   `package --pkg-generator DEB,WHEEL`, `install --dev`, and `list` are tailored to
   Holoscan workflows. In particular, the dev-import hook makes
   `import holoscan.<your_slug>` work from any shell against the live build tree, and
   the packaging command produces ABI-conscious Debian and wheel artifacts ready for an
   APT repo or PyPI. Replicating this from scratch in a generic Python/C++ project is
   significant infrastructure work.

A clean project template gives you none of the above. A Holoscan Module gives you all
three by adopting a small set of conventions (the metadata schema and directory layout
described below).

## 2. Tutorial Prerequisites

Over the next sections we'll walk through creating a dedicated project repository for your Holoscan Module.
Once the project is initialized, you'll need to either build your code via a container or on your local host.
Please install the appropriate dependencies before continuing.

- A local clone of the **HoloHub** repository
- [Python >= 3.10](https://www.python.org/downloads/) on the host machine to run the `holohub` script

### Container Approach (Recommended)

- [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), including the buildx plugin (`docker-buildx-plugin`)
- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (v1.12.2 or later)

### Local Approach

- **Holoscan SDK >= 4.2.0**
- **CMake >= 3.24** and a C++17-capable toolchain (GCC recommended).
- **For C++ modules:** `pybind11-dev`, `libgtest-dev`, `clang-format`.

## 3. Create an External Module

In this step we'll use the Holoscan template to create a self-contained git repository hosted anywhere
you like, such as on GitHub. Any HoloHub-based application or Holoscan SDK project can then declare
and leverage the project as a dependency.

### 3.1 Install dependencies

Navigate to your HoloHub clone, then run the following CLI command to install the minimum template generation dependencies in your
local Python environment:

```bash
./holohub setup --scripts template
```

### 3.2 Scaffold with `./holohub create`

From your HoloHub clone:

```bash
./holohub create holoscan-my-sensor \
    --template modules/template \
    --directory ~/repos
```

Cookiecutter will prompt via command line for project details. Enter the following suggested values
or customize with your own:

| Prompt | Value | Used for |
| --- | --- | --- |
| `project_name` | `My Sensor Holoscan Module` | Display name |
| `module_slug` | `my_sensor` | Python import + C++ namespace |
| `module_repo_name` | `holoscan-my-sensor` | Repo directory + package names |
| `operator_slug` | `my_sensor_op` | Initial operator |
| `language` | `cpp` or `python` | Implementation language |
| `version` | `0.1.0` | Initial semver |
| `holoscan_version` | `4.2.0` | Minimum SDK version |
| `full_name`, `affiliation`, `contact_email` | your details | Authorship and Debian metadata |

The CLI will further derive some names using these rules:

- `module_slug=my_sensor` → C++ namespace `holoscan::my_sensor`, Python package
  `holoscan.my_sensor`, CMake options `OP_my_sensor_op`, `MY_SENSOR_BUILD_TESTING`.
- `operator_slug=my_sensor_op` → C++ class `MySensorOp` (CamelCase from snake_case).
- `module_repo_name=holoscan-my-sensor` → directory + PyPI/Debian package names.

*Advanced Users and AI Agents:* For non-interactive use, pass values directly:

```bash
./holohub create holoscan-my-sensor \
    --template modules/template \
    --directory $HOME \
    --interactive false \
    --context project_name="My Sensor Holoscan Module" \
    --context module_slug=my_sensor \
    --context operator_slug=my_sensor_op \
    --context language=cpp \
    --context full_name="Jane Doe" \
    --context affiliation="Example Corp" \
    --context contact_email=jane@example.com
```

You will see the following output indicating success:

```text
Holoscan Module 'my_sensor' created successfully!

Implement your operator (MySensorOp) in:
  operators/my_sensor_op/my_sensor_op.cpp

Build and run:
  ./holohub run-container
  # Inside the container:
  ./holohub build my_sensor_pipeline
  ./holohub run   my_sensor_pipeline --language python

Git repository initialised. Push to a remote when ready:
  git remote add origin <your-repo-url>
  git push -u origin main

Register your module at https://nvidia-holoscan.github.io/ when ready.
Successfully created new project: holoscan-my-sensor
Directory: /home/myuser/holoscan-my-sensor

Possible next steps:
- Implement your operator in /home/myuser/holoscan-my-sensor/operators/
- Update metadata.json: /home/myuser/holoscan-my-sensor/metadata.json
- Update project README
- Build and test with HoloHub CLI
```

### 3.3 Tour the Generated Tree

Let's take a look at the folder we created in the previous section at `/home/myuser/holoscan-my-sensor`.

```text
holoscan-my-sensor/
├── metadata.json              # Schema urn:holohub:module:v2 (identity, operators,
│                              #   namespace, binary_packages, platforms, SDK pin)
├── pyproject.toml             # scikit-build-core; selectively builds the module
├── CMakeLists.txt             # find_package(holoscan), BUILD_ALL toggle, python staging
├── Dockerfile                 # Pinned to nvcr.io/.../holoscan:<ver>-cuda13-dgpu
├── README.md                  # Module-facing readme (edit me)
├── .clang-format              # C++ modules only
├── .gitignore
├── holohub                    # Wrapper script; first run bootstraps the HoloHub CLI
│                              #   via sparse-checkout into ./.holohub/
├── cmake/                     # HoloHubConfigHelpers, pybind11 integration, deb config
│                              #   (copied in by the cookiecutter post-gen hook)
├── operators/
│   └── my_sensor_op/
│       ├── metadata.json      # Schema urn:holohub:operator:v1
│       ├── CMakeLists.txt
│       ├── my_sensor_op.hpp   # C++ stubs (or my_sensor_op.py for pure-Python modules)
│       ├── my_sensor_op.cpp
│       └── python/            # pybind11 bindings (C++ modules only)
├── applications/
│   └── my_sensor_pipeline/
│       ├── metadata.json
│       ├── CMakeLists.txt
│       └── my_sensor_pipeline.{cpp|py}
├── python/holoscan/my_sensor/
│   └── __init__.py            # Re-exports MySensorOp from the per-operator submodule
├── tests/
│   ├── cpp/                   # GTest stubs (C++ modules only)
│   └── python/
│       ├── conftest.py        # Extends holoscan.__path__ to the build tree
│       ├── pytest.ini
│       └── test_my_sensor_op.py
└── .github/workflows/ci.yml   # Lint, CMake configure, GPU build/test jobs
```

Key things to note:

- `metadata.json` uses schema `urn:holohub:module:v2`. The most important fields for
  discovery and packaging are `name`, `namespace.{cpp,python}`, `operators`,
  `binary_packages.{debian,pypi,install_commands}`, `platforms`, and
  `holoscan_sdk.minimum_required_version`. Update `source_repository` and `authors`
  before publishing.
- `pyproject.toml` uses scikit-build-core. Its `cmake.args` are pre-set to
  `-DMY_SENSOR_BUILD_TESTING=OFF -DBUILD_ALL=OFF -DOP_my_sensor_op=ON` so the wheel
  builds only what the module ships.
- `CMakeLists.txt` calls `find_package(holoscan REQUIRED)`. The
  `BUILD_ALL` option defaults to `ON` when the project is the top-level build and `OFF`
  when nested inside a parent build. `MY_SENSOR_BUILD_TESTING` is a module-scoped
  toggle, independent of CMake's global `BUILD_TESTING`.
- The `holohub` wrapper script in the project root delegates to the HoloHub CLI. On
  first invocation it sparse-checks-out the CLI tools into `./.holohub/`, pinned by
  `CLI_PINNED_COMMIT`.

### 3.4 Implement the Operator

Now that our scaffolding is in place, it's time to implement our custom Holoscan SDK operator.

**C++-based module**: Open `operators/my_sensor_op/my_sensor_op.hpp` and `my_sensor_op.cpp`. The template ships
TODO stubs. A minimal `compute()` that forwards an input tensor to an output port looks
like:

```cpp
// my_sensor_op.cpp
#include "my_sensor_op.hpp"

namespace holoscan::my_sensor {

void MySensorOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<Tensor>>("in");
    spec.output<std::shared_ptr<Tensor>>("out");
}

void MySensorOp::compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext&) {
    auto in = op_input.receive<std::shared_ptr<Tensor>>("in").value();
    // TODO: apply your sensor-specific processing here.
    op_output.emit(in, "out");
}

}  // namespace holoscan::my_sensor
```

**pure-Python module**: add your implementation in `operators/my_sensor_op/my_sensor_op.py`.

**C++/Python modules**: every constructor parameter or method you want exposed in Python
must also be added to the pybind11 trampoline class in
`operators/my_sensor_op/python/_my_sensor_op_bindings.cpp`. The trampoline manually
constructs the `OperatorSpec` (it is not auto-generated from C++ headers). Mirror new
C++ parameters by adding `py::arg("<name>")` entries to the `.def(py::init<...>(), ...)`
call and forwarding the value into the C++ constructor.

Update each `metadata.json` (`operators/my_sensor_op/metadata.json`) with details
about your operator.

The generated `Dockerfile` extends the official Holoscan SDK image with standard build packages
for Holoscan-based projects. For container-based builds, add any custom packages for building and
developing your module before moving on.

### 3.5 Build, Test, and Iterate in the Container

Use the `./holohub` wrapper to drive it:

```bash
cd ~/holoscan-my-sensor

# Single command to build and launch the container, then build and run the demo application
./holohub run my_sensor_pipeline --language <cpp/python>

# Run CTest (C++) and PyTest (Python)
./holohub test

# Launch the development environment for interactive builds and debugging
./holohub run-container
```

Notes:

- `MY_SENSOR_BUILD_TESTING` defaults `ON` when you build the module standalone and `OFF`
  when it is nested under a parent build, so tests run automatically here but a
  downstream consumer that pulls your module via FetchContent does not pay the test
  cost by default.
- The Python test suite uses `SKIP_RETURN_CODE 5` in CTest: when `holoscan` is not
  importable in the current environment, pytest collects zero items and exits 5, which
  CTest treats as **Skipped** rather than failed. This makes the same test invocation
  valid on both GPU and CPU-only hosts.

### 3.6 Use the Live Build Tree from Any Shell

To use the module from a Python shell or notebook outside of the build directory,
install a development hook:

```bash
./holohub install --dev
python -c "import holoscan.my_sensor; print(holoscan.my_sensor.__file__)"
```

The hook writes a `.pth` file plus a small shim into your site-packages that redirects
`holoscan.my_sensor` imports to the live build tree. Re-running `./holohub build` after
a source edit takes effect immediately — no wheel re-install needed. Remove it when
you're done:

```bash
./holohub install --dev --uninstall
```

### 3.7 Declare Dependencies on Other Modules (Optional)

If your module depends on operators from another Holoscan Module, add a `dependencies`
array to your `metadata.json`:

```json
{
  "module": {
    "name": "holoscan-my-sensor",
    "dependencies": [
      {
        "name": "holoscan-other",
        "source": {
          "git_url": "https://github.com/example/holoscan-other",
          "ref": "0123456789abcdef0123456789abcdef01234567"
        },
        "provides_operators": ["other_op"]
      }
    ]
  }
}
```

Note: `ref` should be a 40-character commit SHA. The resolver accepts tags and branches but
  emits a warning — they are mutable and break reproducibility.

### 3.8 Package: Debian and Wheel

Now that we've implemented and tested our module, it's time to package and share it outside
of this repository.

Run the following command to generate Debian and Python packages:

```bash
./holohub package holoscan-my-sensor --pkg-generator DEB,WHEEL
...
CPack: Create package
CPack: - package: /workspace/holohub/holoscan-my-sensor_0.1.0_arm64.deb generated.
...
*** Created holoscan_my_sensor-0.1.0-cp312-cp312-linux_aarch64.whl
Successfully built holoscan_my_sensor-0.1.0-cp312-cp312-linux_aarch64.whl

Wheel output directory: build/dist
```

Important notes:

- `WHEEL` invokes `python -m build --wheel` against `pyproject.toml`. Output goes to
  `build/dist/`.
- `DEB` runs CMake + CPack via the `holohub_configure_deb()` helper baked into the
  template's `CMakeLists.txt`. Update the `Recommends:` / `Depends:` lines in
  `CMakeLists.txt` if your module needs additional system packages.
- The wheel does **not** declare Holoscan SDK as a runtime dependency. Consumers
  install Holoscan SDK matching their CUDA variant separately. State the required
  Holoscan version in your README and in `metadata.json:module.holoscan_sdk`.
- The build and release stack matches the Dockerfile environment. Use the `--docker-opts` flag
  to adjust environment details, such as handling a different Python version.

The Debian and wheel packages are now available for distribution. The Holoscan Module
template infrastructure does not cover the actual distribution of artifacts. We suggest
[PyPI](https://pypi.org/) for Python wheel publishing.

### 3.9 CI/CD: What Ships Out of the Box

The Holoscan Module template scaffolding suggests a few basic GitHub CI hooks for basic coverage.
For best practice, consider extending with more complete build and run coverage of your
project implementation.

`.github/workflows/ci.yml` defines three jobs:

| Job | Runner | What it does |
| --- | --- | --- |
| Lint | ubuntu-latest | `ruff check` (Python) + `clang-format --dry-run -Werror` (C++) |
| CMake configure | ubuntu-latest | Installs Holoscan CPU wheel, runs CMake to validate the build graph without a GPU |
| Build and test | self-hosted GPU | Runs inside the Holoscan container; `ctest` + `pytest`; stages test artifacts |

You can update these TODOs in `ci.yml`:

1. Update the HoloHub commit pin used by the `holohub` wrapper, if you want CI to track a
   specific tested commit rather than `main`.
2. For runtime validation, we suggest setting up a self-hosted GitHub runner with GPU and any other
   hardware requirements, then updating the GPU workflow to target that machine.

### 3.10 Publish and Register

Your module is ready to share with the Holoscan ecosystem! Here are the paths that we recommend
to share your work and seek adoption.

1. **Push the repo.** Create the canonical GitHub (or GitLab, etc.) repository and
   push. Optionally set the `metadata.json:module.source_repository` to the canonical URL so the
   resolver and discovery tooling can find it.
   - Note: for best practice, we suggest pushing code early and often!
   - Note: While we support the Holoscan open source community, you can use this scaffolding to
     create fully private projects as well. No public sharing necessary.
2. **Publish binaries.**
   - Wheel: `python -m twine upload build/dist/*.whl` to PyPI (or a private index).
   - Debian: upload the `.deb` from `build/holoscan-my-sensor/package/` to your APT
     repository.
   - Recommended: The `binary_packages.{debian,pypi}` block in `metadata.json` should match the
     published names exactly. Update `binary_packages.install_commands` to the actual
     end-user install incantation (e.g., `pip install holoscan-my-sensor` or
     `apt install holoscan-my-sensor`).
3. **Register on the Holoscan ecosystem site.** NVIDIA curates the Holoscan landing page with a
   limited set of high-quality community projects. To request review of your project, open a pull
   request in HoloHub to add a pointer entry to your project in the HoloHub `modules/` directory
   and await review from an NVIDIA Holoscan team maintainer. Early communication goes a long way
   towards timely review decisions!

## 4. (Alternative) Add a HoloHub-hosted Module Descriptor

If you've previously added demo operators to HoloHub, you can keep them there and also publish them
as packages with Holoscan Module tooling.

The `holoscan-gstreamer` module is an example of this approach describing the Holoscan GStreamer
Bridge operators as a redistributable collection.

Although there is no template support today, you can follow existing patterns to add a new
HoloHub-hosted Module descriptor in HoloHub.

### 4.1 **Create the descriptor directory** inside the HoloHub clone

You'll add files to this directory to describe how HoloHub operators should be grouped, built, and
packaged to yield your Holoscan Module.

```bash
mkdir -p modules/holoscan-my-sensor
```

### 4.2 **Add `metadata.json`** using schema `urn:holohub:module:v2`

Refer to the schema at [module.schema.json](/utilities/metadata/module.schema.json) to get started.

The key differences from the external-module case:

- Omit `source_repository`. The resolver uses the absence of a `source` block in a
  consumer's dependency entry to detect HoloHub-hosted modules and look them up here.
- Set `operators` to the folder names in the [operators directory](/operators/) you want this module to
  collect.
- Reference `dockerfile` and `documentation.readme` with paths relative to the project root, i.e.:

  ```json
  {
    "dockerfile": "operators/my_sensor/Dockerfile",
    "documentation": { "readme": "operators/my_sensor/README.md" }
  }
  ```

### 4.3. **Register the descriptor in CMake**

Add a one-line `add_holohub_module` call in [modules/CMakeLists.txt](/modules/CMakeLists.txt).
This will tell the HoloHub CMake build system about the module, define the `-DMODULE_<my_name>` flag,
and enable dependency handling for any applications or operators this module covers.

```cmake
add_holohub_module(holoscan-my-sensor OPERATORS my_sensor)
```

### 4.4 (Optional) Enable wheel packaging

Add a `pyproject.toml` description for building your module Python wheel. Modules housed in
HoloHub need simply delegate to the HoloHub CMake build with the module option enabled.

```toml
[build-system]
requires = ["scikit-build-core>=0.10"]
build-backend = "scikit_build_core.build"

[project]
name = "holoscan-my-sensor"
version = "0.1.0"
requires-python = ">=3.10"

[tool.scikit-build]
cmake.source-dir = "../.."
cmake.args = [
    "-DMODULE_holoscan_my_sensor=ON",
    "-DHOLOHUB_BUILD_PYTHON=ON",
]
wheel.packages = []
```

### 4.5 (Optional) Enable Debian packaging

Add a `holohub_configure_deb(...)` call in `modules/my_sensor/CMakeLists.txt` to enable Debian
packaging. See the function description in [cmake/HoloHubConfigHelpers.cmake](/cmake/HoloHubConfigHelpers.cmake)
for parameters, including package dependency handling.

For advanced cases such as various target platforms and development stacks, dynamically update
the Debian metadata before passing it to this call.

### 4.6. **Build and package** from the HoloHub root

Use the HoloHub CLI to run build and packaging operations:

```bash
./holohub build holoscan-my-sensor
./holohub package holoscan-my-sensor --pkg-generator DEB,WHEEL
```

These commands use Docker, CMake, CPack, and scikit-build-core under the hood to carry out
development operations. Use the `--dryrun` flag to view underlying commands without running them.

See `modules/holoscan-gstreamer/metadata.json` and `modules/holoscan-gstreamer/pyproject.toml`
for a working example.

## 5. Reference Card

Fast lookup for repeat use:

| Step | Command | Key file or output |
| --- | --- | --- |
| Scaffold | `./holohub create <repo> --template modules/template --directory <dir>` | `<dir>/<repo>/` |
| Build | `./holohub build <app> --local` | `build/` |
| Test | `./holohub test` | `ctest` + `pytest` output |
| Dev import | `./holohub install --dev` | `.pth` shim in site-packages |
| Package | `./holohub package <name> --pkg-generator DEB,WHEEL` | `build/dist/*.whl`, `build/<name>/package/*.deb` |
| List modules | `./holohub list` | `MODULES:` section |
| Uninstall dev hook | `./holohub install --dev --uninstall` | (removes the shim) |

## 6. Troubleshooting

### **`import holoscan.<my_module>` fails outside the build tree.**

Possible reasons include:

- The Python build output location is not visible to the Python system path (see `PYTHONPATH`)
- `holoscan<=4.2`: The Holoscan SDK wheel does not support importing from paths other than the
Python wheel installation directory

Run the following command to install a `.pth` file in your current Python directory that will point
at your module build location and also patch the Holoscan SDK import search paths.

```bash
./holohub install --dev
```

### **CMake cannot find `holoscan::core`.**

Either run inside `./holohub run-container`,
  or `pip install holoscan==<version>` in your local environment so the SDK CMake config
  is discoverable.

### **`pytest` exits with status 5.**

Zero tests were collected because `holoscan` is not importable in the current environment.
CTest treats this as *Skipped* via `SKIP_RETURN_CODE 5`.

### **The resolver warns about a dependency `ref`.** Pin to a 40-character commit SHA for

  reproducibility. Tags and branches work but are mutable.

## 7. Next Steps

- Read the companion tutorial (forthcoming) on **consuming a Holoscan Module** from a
  HoloHub project or an external Holoscan SDK application.
- Review `modules/holoscan-gstreamer/` as a HoloHub-hosted module reference, including how
  it declares system-package requirements and points its `pyproject.toml` at the
  HoloHub root for selective builds.
- Refer to the full [CLI Reference Guide](/utilities/cli/cli_reference.md).

Happy Holocoding!
