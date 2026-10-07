# Holoscan modules

## Register an existing module

A Holoscan module packages reusable operators and related components for other
applications. Register its external source here; keep its implementation,
`metadata.json`, build files, tests, and packages in the module's repository.
An existing module does not need to be recreated with the CLI to be registered.

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

5. Submit the JSON change in a pull request to the HoloHub registry branch you
   are contributing to. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for the
   full checks. Do not commit `dist/` or the module implementation here.

The Holoscan CLI has no HoloHub registration command. `holoscan list` discovers
projects in a source checkout; it does not submit a registration. CI validates
the record and remote module metadata, then checks CLI discovery. It does not
build the module.
See the [source record reference](../README.md#source-records) for all fields
and the alternative hosted-content source form. A current registration is
[`holoscan-deltacast.json`](holoscan-deltacast.json).

## Create a new module

Work outside this registry. Install [Holoscan CLI 5.x with the `create` extra](https://github.com/nvidia-holoscan/holoscan-cli#installation)
and activate its Python environment. Use a platform supported by the
[Holoscan SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#prerequisites).
For container builds, prepare Docker and NVIDIA Container Toolkit; choose an
SDK image compatible with your GPU and module. Local builds require the SDK,
CUDA toolkit, compiler, and dependencies described by the generated project.

From the directory where you keep source repositories:

```bash
holoscan create my-module --language cpp --interactive false --context holoscan_version=4.6.0
cd holoscan-my-module
holoscan list
holoscan modes my_module_pipeline
```

Replace `4.6.0` with the minimum SDK version you intend to support. Use
`--language python` for a Python-only module. The default template creates a
standalone `holoscan-my-module/` repository layout with module and operator
metadata, build and packaging files, tests, and a `my_module_pipeline` demo.
The CLI version and SDK version are separate choices.

Implement your operators, connect them in the demo application, and replace
the template's placeholder descriptions, author details, repository URL, and
tags. Update the generated `README.md` and `DEVELOPER.md` with requirements,
inputs, outputs, parameters, and expected results. Record SDK versions as
tested only after verifying them. Keep the generated `requirements-cli.txt`
so contributors can use the same CLI version.

## Build, run, test, and package

Run these commands from the module repository, with its CLI environment active.
Building the demo also builds the operators it uses:

```bash
holoscan build my_module_pipeline
holoscan run my_module_pipeline
holoscan test
```

Build and run use the project's development container by default. Configure
its SDK base image as described in the
[CLI configuration guide](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CONFIGURATION.md),
or pass `--base-img` with an appropriate SDK development image. For a native
build with the SDK and dependencies already installed:

```bash
holoscan build my_module_pipeline --local
holoscan run my_module_pipeline --local
holoscan test --local
```

The C++ template also includes a Python demo; select it with
`holoscan run my_module_pipeline --language python`. Test both interfaces you
intend to publish, including a finite run that checks the operator's output.
`holoscan test` uses the project's CTest driver; keep unit tests and meaningful
pipeline tests in the upstream module.

To produce installable distributions from the module root:

```bash
holoscan package holoscan-my-module --pkg-generator DEB,WHEEL
```

Use `--pkg-generator WHEEL` if you only need a Python wheel. Follow the
generated developer guide to check the artifacts in a separate consumer
project before publishing. Keep releases and package hosting upstream, then
register the source or update its existing `source.ref` here.

See the [CLI reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md)
and `holoscan <command> --help` for flags. Build, run, test, and package accept
`--dryrun --verbose` to inspect their commands before execution.
