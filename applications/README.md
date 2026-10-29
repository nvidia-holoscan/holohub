# Holoscan applications

Applications connect Holoscan operators into complete processing pipelines.
Use an existing application as a runnable example or a starting point for your
own workflow. This directory registers sources; application code lives upstream.

## Find and run an application

| Source | What you will find |
| --- | --- |
| [holoscan-applications](https://github.com/nvidia-holoscan/holoscan-applications) | Applications maintained in the dedicated repository. [Registration](holoscan-applications.json). |
| [Archived HoloHub applications](https://github.com/nvidia-holoscan/holohub/tree/archive-4x/applications) | Earlier examples with their original SDK requirements. [Registration](holohub-applications-archive.json). |

For the dedicated applications repository, use Python 3.12 and Git:

```bash
git clone https://github.com/nvidia-holoscan/holoscan-applications.git
cd holoscan-applications
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
holoscan list
```

`holoscan list` prints the available project names. Read the selected
application's README for its supported SDK, GPU, data, models, and devices.
Follow the repository's [build and run instructions](https://github.com/nvidia-holoscan/holoscan-applications#build-and-run)
to prepare Docker, NVIDIA Container Toolkit, and the SDK image, or a native SDK
environment. Installing the CLI alone does not install those dependencies.

## Build, run, and test

From the application's source repository with its environment active, replace
`<application>` with a name from `holoscan list`:

```bash
holoscan modes <application>
holoscan run <application> --dryrun --verbose
holoscan run <application>
```

If the README requires a mode, append one reported by `holoscan modes` to
both run commands. The preview prints the planned commands; the second run
builds and launches the application. Compare the output with the README's
expected result. With the SDK and dependencies on the host, add `--local` to
both commands. Where CTest tests are configured, use `holoscan test <application>`
(with `--local` for native tests). See the
[CLI reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md)
for build-only commands and options.

If setup fails, run `holoscan env-check` and consult the application's
troubleshooting instructions. Report application failures to its issue tracker.

## Create a new application

1. Follow [holoscan-applications' contribution guide](https://github.com/nvidia-holoscan/holoscan-applications/blob/main/CONTRIBUTING.md),
   or develop in your own repository. For changes to an existing application,
   submit the pull request to its owner.
2. Keep the pipeline, configuration, application-specific operators, data
   preparation, and tests together. Package components intended for reuse in
   a [module](../modules/README.md).
3. Add a README with requirements, setup, run commands, and expected output;
   include ownership and licensing. Supply `metadata.json` using the
   [application schema](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/application.schema.json)
   and configure any advertised CLI build, run, and test operations.
4. Test the documented workflow and a finite smoke test on the declared SDK
   and hardware. Publish the tested source revision and keep CI upstream.

`holoscan create` uses a module template by default. To learn application
composition, start with [Holoscan Tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials)
or adapt an application in the source repository.

## Register an existing application

Check this directory's records first. Applications within an already registered
collection need no separate record; update its revision if necessary.
Otherwise, [request registration with kind `application`](https://github.com/nvidia-holoscan/holohub/issues/new?template=registry.yml&kind=application)
after publishing the source and its metadata.

For a single application, use its CLI project name as the registration name
(usually the application directory, not its metadata display name). A collection
has its own stable name. Follow [Contributing](../CONTRIBUTING.md) for form fields,
manual JSON submissions, validation, and maintainer review.

[Back to HoloHub](../README.md)
