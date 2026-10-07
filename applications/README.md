# Holoscan applications

## Register an existing application

Applications combine Holoscan operators into an end-to-end pipeline. Keep the
code, datasets, models, containers, tests, and application metadata in an
external repository; add only a source record here.

1. Check the JSON records in this directory for an existing registration of
   the same repository and path. A source can contain one application or a
   collection of applications. Update its existing record for a new revision.
2. Add `applications/my-application.json` in your HoloHub checkout:

   ```json
   {
     "name": "my-application",
     "kind": "application",
     "issues_url": "https://github.com/example/my-application/issues",
     "source": {
       "type": "git",
       "url": "https://github.com/example/my-application.git",
       "ref": "v0.1.0",
       "path": "."
     }
   }
   ```

   Replace the example values with your published source, preferably using a
   full commit SHA for `ref`. Use `path` for an application subdirectory or
   collection, such as `applications`; keep the clone URL separate from the
   revision and path. `issues_url` is optional. The lowercase registration
   name must match the JSON filename.
3. Verify the upstream application has its own README and metadata describing
   requirements, SDK compatibility, execution modes, and run commands. Follow
   the [application metadata schema](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/application.schema.json).
   Application descriptions and execution settings belong upstream, not in
   this source record.
4. Validate from the HoloHub root:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirement.txt
   .venv/bin/python utilities/validate_registry.py --index dist/sources.json
   .venv/bin/python utilities/validate_sources.py
   ```

5. Submit the JSON change in a pull request to the HoloHub registry branch you
   are contributing to. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full
   checks. Keep generated indexes and application files out of the change.

The CLI has no HoloHub registration command. `holoscan list` lists local
projects; it does not publish them. CI checks declarations, retrieves metadata,
and verifies CLI discovery without building applications. See the
[source record reference](../README.md#source-records) for all fields and
hosted-content sources. [`holoscan-applications.json`](holoscan-applications.json)
shows a registration for an application collection.

## Create a new application

Develop in an external source repository. Install and activate
[Holoscan CLI 5.x](https://github.com/nvidia-holoscan/holoscan-cli#installation).
Prepare the application's SDK version, GPU, data, models, and any capture or
display devices. Container builds need Docker and NVIDIA Container Toolkit;
native builds need a compatible Holoscan SDK and build dependencies.

Start from an application in
[Holoscan applications](https://github.com/nvidia-holoscan/holoscan-applications/tree/main/applications)
or create a standalone project with this layout:

```text
my_app/
├── metadata.json       # Top-level application object
├── CMakeLists.txt      # Configure/build the application and its dependencies
├── README.md           # Setup, data, build/run steps, and expected output
├── Dockerfile          # Development environment for container builds
├── tests/              # Tests and CTest integration
└── ...                 # Python or C++ pipeline, configuration, and resources
```

Implement the pipeline, declare its dependencies and build rules, and describe
its launch command and working directory in `metadata.json`. Use the
[application schema](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/application.schema.json)
and [shared project definitions](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/project.schema.json)
for the metadata contract. Keep SDK versions, platform requirements, and data
download instructions accurate. Document how to recognize a successful run.

`holoscan create` defaults to a **module** template. To scaffold a module with
a demo application, follow the [module guide](../modules/README.md#create-a-new-module).
For an independent application, supply the files above or use an application
Cookiecutter template with `holoscan create my_app --template /path/to/template` and
the name and options documented by that template.

## Build, run, and test

Run from the application's source repository, not this registry:

```bash
holoscan list
holoscan modes my_app
```

Replace `my_app` with the selector printed by `holoscan list`. For a standalone
application, the selector is its directory name, which can differ from the
display name in metadata. A direct registration must use that selector as its
name; a collection registration can have a separate name. For another
checkout, use `holoscan --project-root /path/to/my_app list`.

Build and run in the application's development container:

```bash
holoscan build my_app
holoscan run my_app
```

The Dockerfile and CLI configuration must select a suitable SDK development
image; use `--base-img` for an explicit image override. See
[SDK image configuration](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CONFIGURATION.md).
If the application defines modes, add a mode reported by `holoscan modes`,
for example `holoscan build my_app replay` and `holoscan run my_app replay`
only when a `replay` mode exists. A run builds the application before launch.

With the SDK and dependencies installed on the host, use native execution:

```bash
holoscan build my_app --local
holoscan run my_app --local
```

After configuring the project's CTest driver and tests:

```bash
holoscan test my_app
```

Use `--local` for native tests. `holoscan test` runs CTest; include a finite
smoke test with expected output, alongside tests for reusable operators.
If the upstream repository supplies pre-commit hooks, run `holoscan lint`.
Keep these checks in the application repository's CI, then publish a revision
and register it using the steps above.

Use `holoscan <command> --help` and the
[CLI reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md)
for additional flags. Build, run, and test support `--dryrun --verbose` for
previewing the selected environment and commands.
