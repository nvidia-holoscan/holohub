---
title: Holoscan Platform and Ecosystem
description: >-
  How the Holoscan platform, ecosystem, and SDK fit together: SDK core libraries and
  core modules, external modules, HoloHub, Holoscan CLI, and where to start.
hide:
  - path
---

Holoscan brings together hardware, software libraries, developer tools, and
reusable components for building real-time streaming applications, from
sensor processing to AI inference and visualization. This page explains how
those pieces fit together when developing with Holoscan SDK 5.

**You will learn:**

- How the Holoscan platform, ecosystem, and SDK relate to one another
- How SDK core libraries differ from core modules and external modules
- What HoloHub and Holoscan CLI provide during development
- Where to start for application development, component reuse, or module authoring

## Platform, ecosystem, and SDK

The **Holoscan platform** brings together NVIDIA technologies for real-time
streaming and sensor processing: GPU compute platforms, sensor connectivity,
software libraries, and development tools. The
[SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation)
describes the supported development configurations, and
[Relevant technologies](https://docs.nvidia.com/holoscan/sdk-user-guide/introduction/relevant-technologies)
explains the surrounding software stack.

The **Holoscan ecosystem** includes the platform and the projects built around
it: NVIDIA and community modules, partner integrations, reference applications,
tutorials, and tools. Each project has its own maintainer, compatibility
requirements, release lifecycle, and support policy.

The **Holoscan SDK** is the framework you use to compose, compile, and run
streaming application graphs. It includes the **SDK core libraries** and
**SDK core modules**. An application can use the core libraries directly and
add the modules it needs. The SDK is one part of the wider platform and
ecosystem.

For example, a camera and IMU application uses the SDK to compose operators and
coordinate samples arriving at different rates. A hardware integration supplies
sensor input, while domain modules can provide image processing, inference, or
visualization. The hardware integration and each additional module must be
compatible with the application's SDK version.

## How the software fits together

--8<-- "docs/ecosystem/holoscan-ecosystem-map.svg"

The **core libraries** provide graph composition, compilation and execution,
operator interfaces, typed ports, temporal contracts, payloads, and transport.
HoloMQ is the SDK's transport layer. It is part of the core libraries;
standalone processes can also use its channel client API.

**Modules** add domain capabilities through reusable operators and supporting
libraries. SDK 5 includes three core modules: **Inference**, **Vision**, and
**Visualization**. The
[Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/)
documents their build options, dependencies, and package consumption
instructions.

Sensor connectivity has its own component documentation. For example,
[Holoscan Sensor Bridge](https://docs.nvidia.com/holoscan/sensor-bridge/latest/index.html)
documents sensor-to-host connectivity and its hardware and software setup.
Use each integration's documentation to select compatible hardware and releases.

## Module location, releases, and support

Core and external describe how a module relates to the SDK source and release:

| Module location | Maintained and released | Where to begin |
| --- | --- | --- |
| **Core module** | In the SDK repository, qualified and released with the SDK | The Holoscan Modules section of the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/) |
| **External module** | As a separate project with its own releases and declared SDK compatibility | The module's documentation and repository, discoverable through the [Modules catalog](../modules/index.md) |

Support is a separate distinction. **Supported core modules** follow the SDK
release's support policy. A **supported external module** has its own documented
support and compatibility scope. **Community modules** are maintained by their
community or partner maintainers under the project's stated policy. NVIDIA
experimental or early-access projects can also live outside the SDK; consult
their release documentation for their support status.

A catalog entry or a repository under the NVIDIA GitHub organization does not
by itself establish product support or NVIDIA AI Enterprise coverage. Check the
specific component and release you intend to use. Likewise, a module available
for another SDK major version may need migration before it works with SDK 5.

## HoloHub and Holoscan CLI

[HoloHub](../index.md) is the community repository and discovery site for
applications, operators, modules, tutorials, and benchmarks. Some components
live in HoloHub; others are listed in its catalog and maintained in separate
repositories. Use it to find a working example or reusable component, then
follow that project's compatibility and usage guidance. Your own application
can consume the SDK and modules in its own repository.

[Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli) provides shared
tooling for scaffolding projects and for metadata-driven build, run, and test
workflows. Repositories configure that tooling through their own wrappers;
HoloHub uses `./holohub`. Follow the selected repository's documented entry
point and container configuration. The CLI package is development tooling;
installing it does not install the SDK runtime or choose a compatible SDK image.

## Choose a starting workflow

| I want to... | Start here | Check before proceeding |
| --- | --- | --- |
| Build my first application | [Install the SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation), then follow the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/) | The platform requirements and APIs available in the selected SDK release |
| Run or adapt a reference application | Browse [HoloHub applications](../applications/index.md) and follow the selected application's README | Its SDK version, hardware, model or data requirements, and wrapper commands |
| Add inference, image processing, or visualization | Choose an SDK core module from the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/) | The module's dependencies, API stability, and installation instructions |
| Reuse an external module | Browse the [Modules catalog](../modules/index.md) and follow [Use a Holoscan module](../tutorials/holoscan-modules/use-a-module/README.md) | Its supported SDK versions, platforms, package availability, and support policy |
| Build a reusable module | Follow [Create a Holoscan module](../tutorials/holoscan-modules/create-a-module/README.md) | The template's SDK target, module conventions, and the generated project's instructions |
| Port an existing application | Read the SDK 4.x migration guides in the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/) | Changes to graph authoring, operators, payloads, and module APIs |

The Holoscan SDK User Guide owns SDK concepts, installation, and core-module
documentation. The HoloHub catalog owns component discovery; each component's
documentation owns its detailed setup and usage. Follow those links for current
commands and release-specific requirements.

## Recap

- The platform combines NVIDIA technologies for real-time streaming and sensor
  processing; the ecosystem includes the projects, integrations, and community
  built around them.
- The SDK includes core libraries and core modules. External modules extend it
  from separately maintained projects.
- Module location, compatibility, release availability, and support are distinct
  properties to check when selecting a component.
- HoloHub helps you discover and reuse components. Holoscan CLI supplies shared
  development workflows through project-specific entry points.
- Next: [Install the SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation)
  or explore [HoloHub](../index.md).
