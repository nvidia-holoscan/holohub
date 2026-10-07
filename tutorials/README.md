# Holoscan tutorials

## Register an existing tutorial

Register a walkthrough, notebook, or tutorial collection by pointing to its
external source. Keep the tutorial text, images, code, data instructions, and
metadata with the source owner.

1. Check this directory's JSON records for the same source. A collection can
   have one registration; a new revision updates that record.
2. Add `tutorials/my-tutorial.json` in your HoloHub checkout:

   ```json
   {
     "name": "my-tutorial",
     "kind": "tutorial",
     "issues_url": "https://github.com/example/holoscan-learning/issues",
     "source": {
       "type": "git",
       "url": "https://github.com/example/holoscan-learning.git",
       "ref": "v0.1.0",
       "path": "tutorials/my-tutorial"
     }
   }
   ```

   Replace the example values with your published source. Prefer a full commit
   SHA for `ref`; use `path: "."` when the tutorial is at the repository root.
   `issues_url` is optional. The lowercase `name` must match the JSON filename.
   The path should lead to the tutorial or collection and its `metadata.json`
   descriptors. A direct tutorial's registration name must match its directory
   name, which is its CLI selector.
3. For a hosted tutorial, replace the entire `source` object with an HTTPS
   location serving its metadata:

   ```json
   {
     "type": "url",
     "url": "https://example.org/holoscan/v1/my-tutorial/metadata.json"
   }
   ```

   Use your actual URL, preferably versioned. URL sources have no `ref` or
   `path`. Choose the source form that readers should use, rather than adding
   both forms for the same tutorial. A directory URL must serve `metadata.json`
   at that location; an HTML tutorial page alone does not pass validation.
4. Validate from the HoloHub root:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirement.txt
   .venv/bin/python utilities/validate_registry.py --index dist/sources.json
   .venv/bin/python utilities/validate_sources.py
   ```

5. Submit the JSON change in a pull request to the HoloHub registry branch you
   are contributing to. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for full
   checks. Keep generated indexes and tutorial content outside the change.

There is no Holoscan CLI command to submit a HoloHub registration. CI checks the
record, retrieves tutorial metadata, and verifies CLI discovery without
executing examples. See the [source record reference](../README.md#source-records) for
the complete format and [`holoscan-tutorials.json`](holoscan-tutorials.json)
for a registered collection.

## Create a new tutorial

Author it in an external repository. The CLI has no built-in tutorial template;
create a self-contained directory such as:

```text
tutorials/my-tutorial/
├── README.md           # Learning goal and step-by-step walkthrough
├── metadata.json       # Top-level tutorial object
├── images/             # Diagrams or expected-output screenshots, if useful
└── ...                 # Notebooks, example projects, scripts, and configuration
```

Write the learning goal and intended audience first. Explain the required SDK
version, hardware, environment, datasets, and estimated steps. Include commands,
expected results, and troubleshooting for the parts readers are likely to
encounter. Keep linked files inside the source repository and document any
external downloads and their licenses.

For a source tutorial with `metadata.json`, use the
[tutorial schema](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/tutorial.schema.json)
and its [shared project definitions](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/project.schema.json).
The `tutorial` object includes `name`, `authors`, `version`, `changelog`, `tags`,
`holoscan_sdk`, and `requirements`. Describe the tutorial's requirements and
record the SDK versions you actually tested. Keep this metadata upstream; it
is separate from `tutorials/my-tutorial.json` in this registry.

The [Holoscan tutorials repository](https://github.com/nvidia-holoscan/holoscan-tutorials)
provides examples and authoring guidance. If the tutorial needs a runnable
sample, follow the [application guide](../applications/README.md#create-a-new-application)
or [module guide](../modules/README.md#create-a-new-module) to develop that sample.

## Build and verify the tutorial

A Markdown walkthrough has no compilation step. Notebooks, documentation sites,
and code examples have their own dependencies and execution steps; document
them in the tutorial. Installing the CLI alone does not install the SDK,
notebook runtime, or sample data.

For an accompanying application that supports the Holoscan CLI, install and
activate [Holoscan CLI 5.x](https://github.com/nvidia-holoscan/holoscan-cli#installation),
then run these commands from that application's source root:

```bash
holoscan list
holoscan modes my_app
holoscan build my_app
holoscan run my_app
```

Replace `my_app` with a selector from `holoscan list` and select a mode when
the example requires one. These commands use the application's configured
development container and SDK image. Add `--local` to build and run when the
SDK and dependencies are installed on the host. Use `holoscan test my_app`
when the example provides CTest integration. See the
[CLI reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md)
for configuration and command options.

Use `holoscan` for supported operations in the walkthrough. For notebook or
documentation tooling that the CLI does not provide, give the tool's own
commands explicitly. A tutorial's `metadata.json` alone does not supply an
application build system.

Before publishing, follow the complete walkthrough from a clean environment,
execute notebook cells in order where applicable, and compare results with the
documented output. Run the source repository's metadata and link checks, and
update its version and changelog when the tutorial changes. Publish the source,
then register it or update the existing registration above.
