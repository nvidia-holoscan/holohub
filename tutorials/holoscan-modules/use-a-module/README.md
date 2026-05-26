# Tutorial: Use a Holoscan Module

In this tutorial you will learn how to consume a **Holoscan Module** from a downstream
project built on the Holoscan SDK. We will walk through declaring the dependency,
installing or building the Module, and importing its operators in your own code.

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
beyond a single application or repository. Consumers get a working, namespaced import ready to use
out of the box, without needing to learn the intricate project build details.

```py
# Use
from holoscan.my_sensor import MySensorOp
```

The primary form of a Module is an **External Module** — a standalone git repository with
its own operators, applications, tests, packaging, and CI. Consumers fetch its sources at
build time or install a published binary package.

In this tutorial we use a fictional `holoscan-my-sensor` external Module as the running
worked example.

> For in-tree Module usage (modules that live inside the HoloHub monorepo rather than as
> standalone repositories), see
> [Appendix B](#appendix-b-in-tree-module-dependencies-holohub-subproject).

### Who consumes a Holoscan Module?

A Holoscan Module is the right dependency to declare when:

- **You are building a library or end-user product** on the Holoscan SDK and need
  and want versioned, supported operators rather than vendored or copy-pasted source.
- **You are building a HoloHub sample application** that needs hardware-specific or
  domain-specific operators — for example, a robotics application that needs a vendor
  camera driver, or a medical-imaging app that needs an image-reconstruction library.
- **You are building another Holoscan Module** whose operators depend on operators
  from an upstream Module (a transitive dependency).

In each case the goal is the same: pick up a versioned set of operators by referencing
them in metadata or installing a binary package, without forking source or maintaining
build glue.

### Why use a Holoscan Module instead of copy-pasting operators or rolling your own?

1. **Versioned and supported.** The Module's maintainer owns its build, ABI, test,
   and release. You consume a pinned commit (HoloHub flow) or a pinned package
   version (binary install) and get bug fixes and SDK-compatibility updates for free.
2. **Discoverable.** Curated Holoscan Modules may be listed alongside SDK-supported
   and community work on the NVIDIA HoloHub website. Prospective users and their AI
   agents find your dependency choices by searching the Holoscan ecosystem instead of
   guessing repo names. A vendored copy puts the entire burden of discovery on you.
3. **Drop-in.** The HoloHub resolver fetches and wires Modules into your CMake build
   automatically for HoloHub-flow consumers. Binary consumers get the same operators
   via standard `pip install` or `apt install` plus a single `find_package` (C++) or
   `import` (Python). No vendor-specific build glue.

A Holoscan Module dependency gives you all of the above by adopting a small set of conventions
(the metadata schema and dependency-resolver flow described below).

## 2. Tutorial Prerequisites

This tutorial covers several approaches to consume a Holoscan Module in your project. The base prerequisites apply to all paths;
each path's section below calls out anything extra it needs.

- The target Holoscan Module — its name (e.g., `holoscan-my-sensor`), its source URL
  or its published binary package name, and either a release commit SHA (HoloHub
  flow) or a binary package version (install flows).
- **Holoscan SDK >= 4.2.0** matching the Module's
  `metadata.json:module.holoscan_sdk.minimum_required_version`.
- [Python >= 3.10](https://www.python.org/downloads/) on the host machine to run the
  `holohub` script or your Python application.

### Container Approach (Recommended for HoloHub Flow)

- [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), including the buildx plugin (`docker-buildx-plugin`)
- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (v1.12.2 or later)
- A local clone of the **HoloHub** repository — the `./holohub` CLI lives there.

### Local Approach (for External C++ / Python / Source-Embedding Flows)

- **CMake >= 3.24** and a C++17-capable toolchain (GCC recommended) for C++ and
  source-embedding flows.
- `pip` and a Holoscan SDK wheel installed in the same environment for the Python
  flow.
- Network access during CMake configure for the source-embedding flow.

## 3. Available Approaches

Pick the path that matches your project, then jump to the corresponding section.

| Your project is… | Use… |
| --- | --- |
| A HoloHub application, operator, workflow, or benchmark | [Path A (§4)](#4-path-a--consume-from-a-holohub-subproject-primary) |
| Another Holoscan Module declaring a transitive dependency | [Path A (§4.7)](#47-module-to-module-transitive-dependencies) |
| A standalone C++ application using a module binary package | [Path B (§5)](#5-path-b--external-c-project-binary-install) |
| A pure-Python application or notebook | [Path C (§6)](#6-path-c--external-python-project-pip-install) |
| A standalone C++ project using module sources | [Path D (§7)](#7-path-d--source-level-embedding-in-a-fully-external-project) |
| A HoloHub subproject driven directly from CMake (no `./holohub` CLI) | [Appendix A](#appendix-a-embed-directly-in-holohub-cmake-handling) |
| A HoloHub subproject depending on a module within the HoloHub monorepo | [Appendix B](#appendix-b-in-tree-module-dependencies-holohub-subproject) |

## 4. Path A — Consume from a HoloHub Subproject (Primary)

The canonical flow. The HoloHub CLI's resolver does the work; you just declare the
dependency.

### 4.1 (Optional) Scaffold a HoloHub subproject

If you don't already have a subproject, you can use the HoloHub CLI to create one.
Pick the template that matches the type of project you want to build, then fill in
the interactive prompts with your project details.

```bash
./holohub create my_app --template applications/template
./holohub create my_op  --template operators/template
```

The scaffolded subproject ships a `metadata.json` ready for you to add a
`dependencies.modules[]` block as shown below. Run `./holohub list` to see existing
subprojects, modules, and operators available in your HoloHub clone.

### 4.2 Declare the dependency

Edit your subproject's `metadata.json`. For an application:

```json
{
  "$schema": "urn:holohub:application:v2",
  "application": {
    "name": "my_app",
    "dependencies": {
      "modules": [
        {
          "name": "holoscan-my-sensor",
          "source": {
            "git_url": "https://github.com/example/holoscan-my-sensor.git",
            "ref": "0123456789abcdef0123456789abcdef01234567"
          },
          "provides_operators": ["my_sensor_op"]
        }
      ]
    }
  }
}
```

Field reference (from `utilities/metadata/module.schema.json`, `module_dependency`
definition):

- **`name`** (required) — Module name as published in its own
  `metadata.json:module.name`.
- **`source.{git_url, ref}`** — the Module's git URL and a pinned 40-character commit SHA.
  The resolver warns if `ref` is not a 40-character SHA (mutable tags and branches lose
  reproducibility).
- **`provides_operators`** — operators this dep contributes. Used by the resolver for
  *lazy* fetching: only Modules whose operators are actually enabled (`OP_<name>=ON`)
  get fetched at build time.
- **`version`** (optional) — informational; not enforced by the resolver today.

Those metadata details are used by the HoloHub CLI in the next section.

### 4.3 Build

```bash
./holohub build my_app
```

**How it works:** When you run `./holohub build`, two layers of tooling run to
resolve and lazily fetch any necessary external module sources.

1. The HoloHub CLI scans all `metadata.json` files in the repository to build a metadata database. It reads the `dependencies.modules[]` declarations from your subproject's metadata.
2. For each external module dependency, the CLI writes a `holohub_declare_external_module(...)` entry into a generated CMake manifest file. This function defines a `FetchContent_Declare`
entry for each source URL and tag, along with a HoloHub variable detail to note what
resources the external module provides.
3. CMake includes the manifest during configuration. If any resources from outside HoloHub
are requested, the CMake build uses `FetchContent_MakeAvailable` to lazily clone each declared module's source repository. Only modules whose operators are actually enabled in the build are fetched.

The resolver handles `FetchContent` plumbing automatically — you do not write
`FetchContent_Declare` or `FetchContent_MakeAvailable` calls by hand. Instead, simply
register the operator as a required or optional dependency for your project in `applications/CMakeLists.txt`:

```cmake
add_holohub_application(my_app DEPENDS OPERATORS my_sensor_op)
```

This is what tells CMake which operators to enable (`OP_my_sensor_op=ON`), which in
turn triggers the manifest's lazy fetch for the module that provides them.

### 4.4 Iterate against a local working copy

When iterating against an unreleased dep, override the fetched source with
`HOLOHUB_LOCAL_<UPPER_NAME>` (underscores in place of hyphens, all uppercase):

```bash
export HOLOHUB_LOCAL_HOLOSCAN_MY_SENSOR=/path/to/local/holoscan-my-sensor
./holohub build my_app
```

The resolver emits the local path as a `FETCHCONTENT_SOURCE_DIR_<UPPER>` cache
variable so FetchContent uses your working tree instead of cloning. The `source.ref`
in `metadata.json` is ignored while the env var is set.

### 4.5 Use the operators in your code

Python application:

```python
from holoscan.core import Application
from holoscan.my_sensor import MySensorOp
from holoscan.operators import HolovizOp

class MyApp(Application):
    def compose(self):
        sensor = MySensorOp(self, name="sensor")
        viz = HolovizOp(self, name="viz")
        self.add_flow(sensor, viz, {("out", "receivers")})

if __name__ == "__main__":
    MyApp().run()
```

C++ application: include the public header the Module installs (path is documented in
its README) and link via the Module's exported targets in your HoloHub-app
`CMakeLists.txt`:

```cmake
target_link_libraries(my_app PRIVATE holoscan::my_sensor_op)
```

### 4.6 Discover what's available

```bash
./holohub list   # MODULES: section lists each Module's name, language, operators
```

### 4.7 Module-to-module (transitive) dependencies

If you are building a Holoscan Module that itself depends on another Module, use the
`module.dependencies[]` array in your module's `metadata.json` — it follows the same
schema as the `application.dependencies.modules[]` shown in §4.2 above. Refer to the "Create a Holoscan Module" tutorial for the full schema
reference, the `HOLOHUB_LOCAL_*` override, and the SHA-pinning discipline.

## 5. Path B — External C++ Project (Binary Install)

Projects that live outside HoloHub can easily consume external module binaries
like any other binary package.

### 5.1 Install the binary

Review the Module's `metadata.json:module.binary_packages` for recommended installation commands.
For example, the installation command to add to your Dockerfile or host setup script might install from a public APT repository:

```bash
# Debian / Ubuntu
apt install holoscan-my-sensor
```

### 5.2 CMake integration

In your project's `CMakeLists.txt`:

```cmake
find_package(holoscan REQUIRED COMPONENTS core)
find_package(holoscan-my-sensor REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE
    holoscan::core
    holoscan::my_sensor_op
)
```

Modules are responsible for shipping a CMake package config — for example,
`/usr/lib/<triplet>/cmake/holoscan-my-sensor/holoscan-my-sensor-config.cmake` — that
exports targets under the `holoscan::` namespace. For a working reference, see
`operators/gstreamer/cmake/holoscan-gstreamer-config.cmake.in` and the
`find_package(holoscan-gstreamer)` install-tree test under
`operators/gstreamer/tests/cmake/test_find_package/`.

If the package is not found, ensure the install prefix is on `CMAKE_PREFIX_PATH`, or
that you installed via the system package manager into a default-searched prefix
(typically `/usr`). Report any issues to the appropriate module repository maintainer.

## 6. Path C — External Python Project (Pip Install)

Module wheels available on PyPI can be installed like any other PyPI dependency:

```bash
pip install holoscan              # the SDK matching your CUDA variant
pip install holoscan-my-sensor    # the module
```

Use the operator like any built-in Holoscan operator:

```python
from holoscan.core import Application
from holoscan.my_sensor import MySensorOp
from holoscan.operators import HolovizOp

class MyApp(Application):
    def compose(self):
        self.add_flow(
            MySensorOp(self, name="sensor"),
            HolovizOp(self, name="viz"),
            {("out", "receivers")},
        )

if __name__ == "__main__":
    MyApp().run()
```

The wheel installs the operator into a Python package named after the Module — for
`holoscan-my-sensor` that is `holoscan.my_sensor` — alongside the SDK's own
`holoscan.operators`, `holoscan.core`, etc. The Module's
`python/holoscan/my_sensor/__init__.py` re-exports the per-operator submodules so a
single import is enough.

> **Version note:** The Holoscan SDK wheel for versions 4.2 and earlier does not
> support importing packages from paths outside its own wheel installation directory.
> We recommend Holoscan SDK >= 4.3 for importing from the `holoscan` namespace.

## 7. Path D — Source-Level Embedding in a Fully External Project

Projects that rely directly on CMake without HoloHub CLI tooling can leverage
CMake's [`FetchContent`](<https://cmake.org/cmake/help/latest/module/FetchContent.html>) function to fetch and build open-source Holoscan Modules
directly.

```cmake
include(FetchContent)
FetchContent_Declare(holoscan_my_sensor
    GIT_REPOSITORY https://github.com/example/holoscan-my-sensor.git
    GIT_TAG        0123456789abcdef0123456789abcdef01234567
)
FetchContent_MakeAvailable(holoscan_my_sensor)

target_link_libraries(my_app PRIVATE holoscan::my_sensor_op)
```

Building from source may require additional dependency setup and custom configuration
options. Please refer to the individual Module's guidance on requirements
and best practices. A binary install (Paths B and C) is usually a simpler choice for fully external projects, when available.

## 8. Reference Card

| Goal | Where or what |
| --- | --- |
| Declare a Module dep (HoloHub app) | `metadata.json:application.dependencies.modules[]` |
| Declare a transitive dep (Module → Module) | `metadata.json:module.dependencies[]` |
| External Module reference | Include `source.{git_url, ref}` (40-char SHA) and `provides_operators` |
| Local override | `HOLOHUB_LOCAL_<UPPER_NAME>=<path>` env var |
| Build a HoloHub subproject with deps | `./holohub build <app>` |
| Install binary (C++) | `apt install holoscan-<name>` |
| Install binary (Python) | `pip install holoscan-<name>` |
| Use from C++ | `find_package(holoscan-<name> REQUIRED)` + `target_link_libraries(... holoscan::<target>)` |
| Use from Python | `from holoscan.my_module import MyOp` (substitute your Module name) |
| Embed via source — HoloHub tree, CMake-direct (Appendix A) | `holohub_declare_external_module(...)` + `add_holohub_application(... DEPENDS OPERATORS …)` — root build's post-step fetches |
| Embed via source — fully external (Path D) | `FetchContent_Declare(...)` + `FetchContent_MakeAvailable(...)` |
| Discover Modules | `./holohub list` |

## 9. Troubleshooting

- **`find_package(holoscan-my-sensor)` fails.** Confirm install:
  `dpkg -L holoscan-my-sensor | grep cmake`. Add the prefix to `CMAKE_PREFIX_PATH`
  if it is not in a standard location.
- **`import holoscan.my_sensor` raises `ModuleNotFoundError`.** Check that
  `pip show holoscan-my-sensor` succeeds in the same Python environment, and that the
  Holoscan SDK wheel is installed in that same env.
- **Resolver warns "ref is not a 40-character commit SHA".** Replace the tag or branch
  in `source.ref` with the commit SHA it resolves to.
- **`HOLOHUB_LOCAL_*` override has no effect.** Naming: take the Module name, replace
  hyphens with underscores, uppercase. `holoscan-my-sensor` →
  `HOLOHUB_LOCAL_HOLOSCAN_MY_SENSOR`.
- **The operator is not built even though you declared it.** Under FetchContent +
  `PROJECT_IS_TOP_LEVEL` defaults, the Module's `BUILD_ALL` is `OFF`, so only the
  operators marked with `OP_<op>=ON` get built. In a HoloHub tree (Path A or
  [Appendix A](#appendix-a-embed-directly-in-holohub-cmake-handling)),
  list the operator in `add_holohub_application(... DEPENDS OPERATORS …)` — the
  helper sets the flag for you. In a fully external CMake build (Path D), pass
  `-DOP_<op>=ON` on the configure line or `set(...)` it before
  `FetchContent_MakeAvailable`.
- **HoloHub build fails with "No content recorded" inside
  `FetchContent_MakeAvailable`.** The external operators manifest is stale. Delete
  `build/external_operators_manifest.cmake` and reconfigure; the resolver regenerates
  it on each build.

## 10. Next Steps

For the full CLI reference, see `utilities/cli/cli_reference.md`.

Happy holocoding!

---

## Appendix A: Embed directly in HoloHub CMake handling

For developers working inside a HoloHub subproject who need to declare a Module
dependency from CMake directly, bypassing the `./holohub` CLI / `metadata.json` flow
of Path A. This approach is documented for advanced edge cases only; most users should favor
Path A with HoloHub CLI tooling when contributing to HoloHub.

Reuse HoloHub's existing CMake helpers — `holohub_declare_external_module()` and
`add_holohub_application()` — from your subproject's `CMakeLists.txt`. Together they
register the FetchContent declaration, record the operator-to-provider mapping
(`HOLOHUB_EXT_OP_<op>_PROVIDER`), and force-enable the operators your application
needs. You should not call `FetchContent_MakeAvailable` yourself: the HoloHub root
`CMakeLists.txt` has a post-step that walks the `*_PROVIDER` variables and calls
`FetchContent_MakeAvailable` once per provider whose operators are enabled.

```cmake
# Somewhere in your HoloHub subproject's CMakeLists.txt
holohub_declare_external_module(holoscan_my_sensor
    GIT_REPOSITORY https://github.com/example/holoscan-my-sensor.git
    GIT_TAG        0123456789abcdef0123456789abcdef01234567
    PROVIDES_OPERATORS my_sensor_op
)

# add_holohub_application implicitly force-enables OP_<op>=ON for every operator
# listed under DEPENDS OPERATORS — you should not set those cache variables yourself.
add_holohub_application(my_app DEPENDS OPERATORS my_sensor_op)
```

The actual link against the Module's exported target goes in your application's
subdirectory `CMakeLists.txt` (the file `add_holohub_application` adds via
`add_subdirectory(my_app)`):

```cmake
# my_app/CMakeLists.txt
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE holoscan::my_sensor_op)
```

Notes:

- The first positional argument to `holohub_declare_external_module` is a
  CMake-identifier-safe provider id (hyphens are replaced with underscores by
  convention). The remaining keyword args (`GIT_REPOSITORY`, `GIT_TAG`, etc.) are
  forwarded to `FetchContent_Declare`.
- The Module's generated root `CMakeLists.txt` detects nested builds via
  `PROJECT_IS_TOP_LEVEL` and defaults `BUILD_ALL=OFF`; only the operators you list
  in `add_holohub_application(... DEPENDS OPERATORS …)` are built. Tests are
  similarly off by default (`<MODULE_NAME>_BUILD_TESTING=OFF`).
- Compared to Path A (the CLI-driven flow), this path is equivalent in build
  output — the CLI's resolver emits the same `holohub_declare_external_module(...)`
  call into a generated manifest. Choose Path A if you want declarative metadata
  and the rest of the CLI's lifecycle commands; choose this path if you want the
  dependency expressed directly in CMake.

---

## Appendix B: In-tree Module Dependencies (HoloHub Subproject)

An **in-tree Module** is a thin descriptor inside the HoloHub monorepo whose operator
sources live in `operators/<name>/` rather than a separate git repository. See
`modules/holoscan-gstreamer/` for the canonical reference. *This path is documented
primarily for legacy operator contributions and is not recommended for most new
operator contributions.*

To consume an in-tree Module from a HoloHub subproject (Path A), omit the `source` block
in `metadata.json`. The resolver looks the Module up under `modules/<name>/metadata.json`
and uses the in-tree sources directly.

```json
{
  "application": {
    "name": "my_app",
    "dependencies": {
      "modules": [
        { "name": "holoscan-gstreamer", "provides_operators": ["gstreamer"] }
      ]
    }
  }
}
```

The `source.{git_url, ref}` field is required only for external Modules; omit it entirely
for in-tree Modules. Everything else — the build command (`./holohub build my_app`), the
local-override mechanism (`HOLOHUB_LOCAL_*`), and the operator import — is identical to
the external-Module flow in Path A.

See `modules/holoscan-gstreamer/metadata.json` for a live in-tree Module you can add as
a dependency to verify your tooling end-to-end.
