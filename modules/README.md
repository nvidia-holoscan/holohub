# Holoscan modules

## Register an existing module

A Holoscan module packages reusable operators and related components for other
applications. Register its external source here; keep its implementation,
`metadata.json`, build files, tests, and packages in the module's repository.
An existing module does not need to be recreated with the CLI to be registered.

To register without editing JSON, [open a registration request](https://github.com/nvidia-holoscan/holohub/issues/new?template=registry.yml&kind=module);
a bot opens the pull request. To edit the record yourself:

1. Check the JSON records in this directory for the same repository and source
   path. A module containing several operators needs one source registration,
   not one per operator or release. Update an existing record for a new revision.
2. Add `modules/my-module.json` in your HoloHub checkout, using this example:

   ```json
   {
     "name": "my-module",
     "kind": "module",
     "issues_url": "https://github.com/example/holoscan-my-module/issues",
     "source": {
       "type": "git",
       "url": "https://github.com/example/holoscan-my-module.git",
       "ref": "v0.1.0",
       "path": "."
     }
   }
   ```

   Replace the example URL and revision with your published source. Prefer a
   full commit SHA for reproducibility. `path` selects the module directory
   inside the repository; `"."` means its root. `issues_url` is optional.
   The lowercase registration name must match the JSON filename and the
   module's declared name, which is its CLI project selector.
3. Confirm that the referenced directory contains the module's own metadata
   and documentation. Describe its operators, SDK compatibility, dependencies,
   namespaces, and binary packages there, using the
   [module metadata schema](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/module.schema.json).
   Those fields do not belong in the source record above.
4. From the HoloHub root, validate and generate the source index:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirement.txt
   .venv/bin/python utilities/validate_registry.py --index dist/sources.json
   .venv/bin/python utilities/validate_sources.py
   ```

5. Submit the JSON change in a pull
   request to the HoloHub registry branch you are contributing to. Follow
   [CONTRIBUTING.md](../CONTRIBUTING.md) for the full checks. Do not commit
   `dist/` or the module implementation here.

The Holoscan CLI has no HoloHub registration command. `holoscan list` discovers
projects in a source checkout; it does not submit a registration. CI validates
the record and remote module metadata, then checks CLI discovery. It does not
build the module.
See the [source record reference](../CONTRIBUTING.md#source-records) for all fields
and the alternative hosted-content source form. A current registration is
[`holoscan-deltacast.json`](holoscan-deltacast.json).

## Create a new module

Follow the [Holoscan Modules tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules),
starting with [Create a Holoscan Module](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/create-a-module).
It walks through scaffolding a standalone repository, implementing a typed
C++ operator, composing and compiling a finite graph, testing its output,
and producing a Debian package. The companion
[Use a Holoscan Module](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/use-a-module)
covers dependencies and consuming a module in another project.

The quick start below targets the Holoscan SDK and Holoscan CLI.
Work outside this registry and install the CLI's `create` extra in a Python
3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'holoscan-cli[create]==5.0.0'
```

Use a platform supported by the [Holoscan SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation).
Container builds require Docker, Docker Buildx, and NVIDIA Container Toolkit.
Native builds require the SDK, CUDA Toolkit 13.x, CMake 4.0 or later, a
C++20-capable compiler, and the generated project's test dependencies.

From the directory where you keep source repositories:

```bash
holoscan create my-module --language cpp --interactive false --context holoscan_version=5.0.0
cd holoscan-my-module
holoscan list
holoscan modes my_module_pipeline
```

The template supports `--language cpp` and creates a standalone
`holoscan-my-module/` repository with module and operator metadata, CMake build
files, a development container, tests, Debian packaging, and a finite
`my_module_pipeline` demo. This scaffold does not generate Python operators,
bindings, or wheels. The CLI and SDK versions are independent;
`holoscan_version=5.0.0` explicitly sets the module's minimum SDK version.

Implement your operators, connect them in the demo application, and replace
the template's placeholder descriptions, author details, repository URL, and
tags. Update the generated `README.md` and `DEVELOPER.md` with requirements,
inputs, outputs, parameters, and expected results. Record SDK versions as
tested only after verifying them. Keep the CLI pins in the generated
`requirements-cli.txt` and `pyproject.toml` synchronized for contributors and CI.

## Build, run, test, and package

Run these commands from the module repository, with its CLI environment active.
Building the demo also builds the operators it uses:

```bash
holoscan build my_module_pipeline
holoscan run my_module_pipeline
holoscan test
```

Build and run use the project's development container by default. Select an
SDK CUDA 13 development image as described in the
[CLI configuration guide](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CONFIGURATION.md),
or pass `--base-img` with an appropriate SDK development image. For a native
build with the SDK and dependencies already installed:

```bash
holoscan build my_module_pipeline --local
holoscan run my_module_pipeline --local
holoscan test --local
```

`holoscan test` uses the project's CTest driver. The tutorial explains
compile-plan tests and deterministic graph tests with `SyntheticClock`. Keep
unit tests and a finite pipeline test that checks the operator's output in the
upstream module.

To produce installable distributions from the module root:

```bash
holoscan package holoscan-my-module --pkg-generator DEB
```

Inspect the Debian package's metadata and contents, then follow the generated
developer guide and the consumption tutorial to install it in a separate
consumer project before publishing. Keep releases and package hosting upstream,
then register the source or update its existing `source.ref` here.

See the [CLI reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md)
and `holoscan <command> --help` for flags. Build, run, test, and package accept
`--dryrun --verbose` to inspect their commands before execution.
