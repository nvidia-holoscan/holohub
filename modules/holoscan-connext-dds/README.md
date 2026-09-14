# RTI Connext DDS for NVIDIA Holoscan

Bring the power of RTI Connext to NVIDIA Holoscan applications. This module
makes it easy for Holoscan applications to publish and subscribe to strongly
typed data using the DDS publish/subscribe communication model, without
requiring applications to build and maintain custom point-to-point
connectivity.

Use it to:

- Connect independent Holoscan applications and non-Holoscan systems using a
  standard publish/subscribe communication model.
- Reuse existing DDS data models or define application-owned data types using
  IDL.
- Decouple producers and consumers, allowing applications to evolve
  independently.
- Control how each data flow is delivered using configurable DDS QoS policies.
- Scale from local processes to distributed systems while using the same
  data-centric communication model.

## Why RTI Connext?

[RTI Connext](https://www.rti.com/products/third-party-integrations/nvidia) is a
real-time data-streaming platform for intelligent distributed systems. It
enables applications to share the right data at the right time, wherever they
run.

Connext is designed for systems that require reliable, low-latency, and
scalable data exchange. Built on the Object Management Group (OMG) Data
Distribution Service (DDS) standard, Connext provides:

- Automatic discovery of compatible publishers and subscribers.
- Direct, data-centric communication without an application-level message
  broker.
- Fine-grained QoS control for each data flow.
- Strongly typed interfaces defined with OMG IDL 4 and DDS-XTypes, including
  support for mutable and extensible types.
- A modular architecture that keeps applications decoupled and easier to
  evolve.

## What can you do with this module?

The module includes generic Python publish and subscribe operators for
application-owned DDS types, C++ video operators, an RTI Shapes Demo
subscriber, and ready-to-run example applications. Everything is built,
tested, and run through HoloHub, with no Connext or Holoscan installation
required on the host.

Each operator documents its own ports, parameters, QoS behavior, memory model,
and limitations. The module README intentionally contains only shared setup and
the recommended starting workflows.

## Components

| Component | Purpose |
| --- | --- |
| [Python publisher](../../operators/dds/dds_pubsub/publisher/README.md) | Publish application-owned DDS types from a Holoscan graph |
| [Python subscriber](../../operators/dds/dds_pubsub/subscriber/README.md) | Emit application-owned DDS types into a Holoscan graph |
| [DDS base](../../operators/dds/base/README.md) | Shared C++ participant and QoS configuration |
| [Shapes subscriber](../../operators/dds/dds_shapes_subscriber/README.md) | RTI Shapes Demo subscriber behavior and limitations |
| [Video publisher](../../operators/dds/video/dds_video_publisher/README.md) | Holoscan RGBA buffers to DDS video samples |
| [Video subscriber](../../operators/dds/video/dds_video_subscriber/README.md) | DDS video samples to Holoscan RGBA buffers |
| [DDS applications](../../applications/dds/README.md) | Python IDL examples and the C++ DDS video application |
| [DDS video application](../../applications/dds/dds_video/README.md) | Camera or synthetic video transported through DDS |
| [Module Dockerfile](Dockerfile) | Holoscan, Connext, Python API, and build environment |
| [Module metadata](metadata.json) | Module version, dependencies, namespace, and subprojects |

## Supported versions

| Component | Version |
| --- | --- |
| Module release | `1.0.0` |
| RTI Connext DDS | `7.7.0` |
| Connext Python API | `rti.connext==7.7.0` |
| NVIDIA Holoscan SDK | `4.6.0` |
| HoloHub container base | Holoscan 4.6 with CUDA 12 |
| Architectures | `x86_64`, `aarch64` |

The module version describes this integration's own API and implementation.
The Connext version is pinned separately in the Dockerfile and metadata. The
Docker build argument `RTI_CONNEXT_VERSION` defaults to `7.7.0`; changing it
requires updating the APT package, Python package, `NDDSHOME`, license mount
path, metadata, and validation together.

## Requirements

- Docker and the NVIDIA Container Runtime supported by HoloHub.
- A system supported by the HoloHub 4.6 container workflow, such as an NVIDIA
  IGX development system or a compatible discrete-GPU host.
- Network access to NVIDIA and RTI package registries during the first image
  build.
- A valid RTI Connext license mounted into the container at runtime.
- Compatible topic names, types, and QoS on communicating endpoints in the
  same DDS domain.
- A V4L2-compatible camera only when using the DDS video camera source. The
  synthetic video source does not require a camera.

Do not install Holoscan, Connext, or the Connext Python API on the host for this
workflow. All build, test, and runtime dependencies are provided inside Docker.

## RTI Connext license

Request and download an activation key from the
[RTI Connext license page](https://content.rti.com/l/983311/2025-07-25/q6729c),
then save it as `rti_license.dat` in the HoloHub repository root:

```bash
curl -L https://content.rti.com/l/983311/2025-07-25/q6729c -o rti_license.dat
```

The filename is ignored by Git and must not be committed. The module container
sets `RTI_LICENSE_FILE` to the Connext 7.7.0 installation path; the HoloHub run
commands below mount the local file at that location.

## Quick start: round-trip validation

From the HoloHub repository root, build and run the self-contained example:

```bash
./holohub build connext_dds_roundtrip --language python
./holohub run connext_dds_roundtrip --language python \
  --docker-opts="-v ${PWD}/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

A successful run publishes and receives two different DDS data types and ends
with:

```text
DDS round trip succeeded: Telemetry=20, Command=20 samples
```

HoloHub reads the module Dockerfile from metadata. Do not pass `--local` or
`--docker-file` to these commands. In `tcsh`, use `${cwd}` instead of `${PWD}`
for the repository path.

## Two independent Holoscan applications

To demonstrate Connext databus communication across process and container
boundaries, open two terminals in the HoloHub repository root. Start the
subscriber first.

Terminal 1:

```bash
./holohub run connext_dds_subscriber --language python \
  --docker-opts="-v ${PWD}/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

Terminal 2:

```bash
./holohub run connext_dds_publisher --language python \
  --docker-opts="-v ${PWD}/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

The publisher and subscriber are separate Holoscan applications running in
separate Docker containers. HoloHub provides host networking, allowing Connext
databus discovery and data exchange. Both applications validate 20 `Telemetry`
and 20 `Command` samples before exiting.

For communication across different hosts, the network must permit the DDS
discovery and user-data traffic selected by the deployment's Connext transport
and discovery configuration.

## Operator configuration

Use the operator links in the [component table](#components) for ports,
arguments, application-owned IDL integration, CMake-driven type generation,
QoS behavior, memory behavior, tests, and operator-specific limitations.

## Build and test

Build the complete module for either supported language through HoloHub:

```bash
./holohub build holoscan-connext-dds --language python
./holohub build holoscan-connext-dds --language cpp
```

Tests and runnable validation commands are documented with the component they
exercise. For example, run the generic Python operator tests with:

```bash
./holohub test dds_pubsub --language python
```

See the [DDS video application](../../applications/dds/dds_video/README.md) for
the headless DDS round-trip, synthetic input, camera, and display commands.

The reference workflow is container-only. A local installation must reproduce
the matching Holoscan SDK, Connext 7.7.0 runtime, Connext Python API, license,
and environment configuration and is outside this module's documented setup.

## Ownership and contact

Vendor: RTI Real-Time Innovations

Contact: `holoscan@rti.com`

## Learn more

- [Connext Developer's Guide](https://community.rti.com/static/documentation/developers/)
- [RTI Connext Professional](https://www.rti.com/products/connext-professional)
- [RTI and NVIDIA](https://www.rti.com/products/third-party-integrations/nvidia)
- [The Connext databus](https://www.rti.com/products/what-is-a-databus)
- [The DDS standard](https://www.rti.com/products/dds-standard)
