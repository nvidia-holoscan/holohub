# HoloHub

HoloHub is the registry for the [NVIDIA Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk)
ecosystem. Use it to find applications, reusable modules, tutorials, benchmarks,
and agent skills for building accelerated sensor-processing and AI pipelines.

HoloHub connects you to projects maintained in their own repositories.
Applications, Tutorials, and Benchmarks are independent repositories: each
provides its own content, requirements, and instructions, and can be cloned and
used directly.

## Explore the ecosystem

| What you want to do | Start here |
| --- | --- |
| Run an application or adapt it for your own workflow | [Holoscan Applications](https://github.com/nvidia-holoscan/holoscan-applications) |
| Learn a concept or follow a development walkthrough | [Holoscan Tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials) |
| Measure performance or read published results | [Holoscan Benchmarks](https://github.com/nvidia-holoscan/holoscan-benchmarks) |
| Find reusable operators packaged as modules | [Module sources](modules/) |
| Find instructions and resources for coding agents | [Skill sources](skills/) |

Browse all registered [application](applications/), [tutorial](tutorials/),
and [benchmark](benchmarks/) sources for additional projects.

## Use a project

1. Choose a project or registered source. A source record points to its location;
   for Git sources, it also identifies the revision and directory to use.
2. Open the source repository and read the project's README. Check its supported
   SDK versions, platforms, hardware, dependencies, and data requirements.
3. Follow that project's setup and build or run instructions from its own
   checkout. Projects that use the [Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli)
   document how to install it and select their applications or modes.

The SDK and hardware requirements vary by project. The
[Holoscan SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation)
covers SDK setup; use the version required by your selected project.

## Earlier HoloHub examples

The `archive-4x` branch preserves the original code collection and is
**closed to new contributions**.

Older guides that invoke `./holohub` refer to the former HoloHub source
collection. Use the release or commit containing the example and launcher.
The current `main` branch is the registry, so build and run instructions for
that earlier collection require a separate checkout.

## Contribute or get help

Contribute application, tutorial, and benchmark changes in their respective
repositories. Maintain reusable modules in their own repositories; see the
[module guide](modules/README.md) to get started. To add or update a project
registration in HoloHub, follow [CONTRIBUTING.md](CONTRIBUTING.md).

Report project-specific problems to that project's issue tracker, using the
source record's `issues_url` when provided. Report missing or incorrect
registrations in [HoloHub issues](https://github.com/nvidia-holoscan/holohub/issues).
