# RTI Connext DDS for NVIDIA Holoscan

Bring the power of RTI Connext to NVIDIA Holoscan applications. This module
makes it easy to publish and subscribe to real-time, strongly typed data from a
Holoscan graph while keeping the data model and communication behavior under
application control.

Use it to connect independent Holoscan applications, reuse an existing DDS data
model, or add new data flows without building custom connectivity for every
application.

## Why RTI Connext?

[RTI Connext](https://www.rti.com/products/connext-professional) is RTI's
real-time data-streaming platform for intelligent distributed systems. It
enables applications to share the right data at the right time, wherever they
run.

Connext is designed for systems that require reliable, low-latency, and
scalable data exchange. Its data-centric architecture allows developers to
focus on the data their applications produce and consume, while configurable
QoS policies control how each data flow is delivered.

Built on the Object Management Group (OMG) Data Distribution Service (DDS)
standard, Connext provides:

- Automatic discovery of compatible publishers and subscribers.
- Direct, data-centric communication without an application-level message
  broker.
- Fine-grained QoS control for each data flow.
- Strongly typed interfaces defined with IDL.
- A modular architecture that keeps applications decoupled and easier to
  evolve.

## What can you do with this module?

The module provides two reusable Python operators in the `holohub.dds`
namespace:

- `ConnextDDSPublisherOp`, which publishes samples received on a Holoscan input
  port.
- `ConnextDDSSubscriberOp`, which emits received DDS samples through a Holoscan
  output port.

Applications supply their own DDS types using Python `rti.idl` declarations or
classes generated from existing IDL files. They also select the domain, topic
names, and XML QoS configuration, allowing the same operators to be reused
across different data models and workflows.

Ready-to-run examples show both a self-contained round trip and communication
between two Holoscan applications in separate Docker containers. Everything is
built, tested, and run through HoloHub, with no Connext or Holoscan installation
required on the host.

## Components

| Component | Purpose |
| --- | --- |
| [DDS operators](../../operators/dds/dds_pubsub/README.md) | Generic Python publisher and subscriber API |
| [DDS applications](../../applications/dds/README.md) | Round-trip and two-process examples using two IDL types |
| [Module Dockerfile](Dockerfile) | Holoscan, Connext, Python API, and build environment |
| [Module metadata](metadata.json) | Module version, dependencies, namespace, and subprojects |

## Supported versions

| Component | Version |
| --- | --- |
| Module release | `1.0.0` |
| RTI Connext DDS | `7.7.0` |
| Connext Python API | `rti.connext==7.7.0` |
| NVIDIA Holoscan SDK | `4.5.0` |
| HoloHub container base | Holoscan 4.5 with CUDA 12 |
| Architectures | `x86_64`, `aarch64` |

The module version describes this integration's own API and implementation.
The Connext version is pinned separately in the Dockerfile and metadata. The
Docker build argument `RTI_CONNEXT_VERSION` defaults to `7.7.0`; changing it
requires updating the APT package, Python package, `NDDSHOME`, license mount
path, metadata, and validation together.

## Requirements

- Docker and the NVIDIA Container Runtime supported by HoloHub.
- A system supported by the HoloHub 4.5 container workflow, such as an NVIDIA
  IGX development system or a compatible discrete-GPU host.
- Network access to NVIDIA and RTI package registries during the first image
  build.
- A valid RTI Connext license mounted into the container at runtime.
- Compatible DDS domain IDs, topic names, types, and QoS on communicating
  endpoints.

Do not install Holoscan, Connext, or the Connext Python API on the host for this
workflow. All build, test, and runtime dependencies are provided inside Docker.

## RTI Connext license

The Apache-2.0 source code in this module does not include an RTI Connext
runtime license. Request and download an activation key from the
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

A successful run publishes and receives two different DDS types and ends with:

```text
DDS round trip succeeded: Telemetry=20, Command=20 samples
```

HoloHub reads the module Dockerfile from metadata. Do not pass `--local` or
`--docker-file` to these commands. In `tcsh`, `${cwd}` can be used instead of
`${PWD}` for the repository path.

## Two independent Holoscan applications

To demonstrate DDS communication across process and container boundaries, open
two terminals in the HoloHub repository root. Start the subscriber first.

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
discovery and data exchange. Both applications validate 20 `Telemetry` and 20
`Command` samples before exiting.

For communication across different hosts, the network must permit the DDS
discovery and user-data traffic selected by the deployment's Connext transport
and discovery configuration.

## Use your own IDL

The operators accept any Python type supported by the Connext Python API. Keep
the IDL and generated code with the application or workflow that owns the data
contract.

For an inline type:

```python
from rti import idl


@idl.struct
class RobotState:
    sequence: int = 0
    position: float = 0.0
```

For an existing IDL file, integrate RTI Code Generator into the application's
CMake build. For example, place this type in `RobotState.idl`:

```idl
struct RobotState {
    int64 sample_id;
    double position;
};
```

The following application `CMakeLists.txt` generates `RobotState.py` whenever
the IDL changes and installs it beside the application:

```cmake
cmake_minimum_required(VERSION 3.20)
project(robot_state_example LANGUAGES NONE)

set(IDL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RobotState.idl")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GENERATED_TYPE "${GENERATED_DIR}/RobotState.py")
set(RTIDDSGEN_HOME "${CMAKE_CURRENT_BINARY_DIR}/rti_home")

# Keep Code Generator state in the build tree and do not copy RTI examples.
file(MAKE_DIRECTORY "${RTIDDSGEN_HOME}/.rti")
file(WRITE "${RTIDDSGEN_HOME}/.rti/rticommon_config.sh"
  "copy_workspace=false\n"
)

find_program(RTIDDSGEN_EXECUTABLE
  NAMES rtiddsgen
  HINTS "$ENV{NDDSHOME}/bin"
  REQUIRED
)

add_custom_command(
  OUTPUT "${GENERATED_TYPE}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${GENERATED_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E env "HOME=${RTIDDSGEN_HOME}"
          "${RTIDDSGEN_EXECUTABLE}"
          -language python
          -replace
          -d "${GENERATED_DIR}"
          "${IDL_FILE}"
  DEPENDS "${IDL_FILE}"
  COMMENT "Generating Python DDS types from RobotState.idl"
  VERBATIM
)

add_custom_target(robot_state_example ALL
  DEPENDS dds_pubsub_python "${GENERATED_TYPE}"
)

install(
  FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/robot_state_example.py"
    "${GENERATED_TYPE}"
  DESTINATION "examples/robot_state_example"
  COMPONENT "robot_state_example-py"
)
```

The module container provides both `rtiddsgen` and `NDDSHOME`, so generation
happens as part of the normal HoloHub container build:

```bash
./holohub build robot_state_example --language python
```

Import the generated module in the application and pass its type to either
operator:

```python
import RobotState as robot_state_idl

from holohub.dds import ConnextDDSPublisherOp, ConnextDDSSubscriberOp

publisher = ConnextDDSPublisherOp(
    fragment,
    domain_id=42,
    topic="RobotState",
    topic_type=robot_state_idl.RobotState,
    name="robot_state_publisher",
)

subscriber = ConnextDDSSubscriberOp(
    fragment,
    domain_id=42,
    topic="RobotState",
    topic_type=robot_state_idl.RobotState,
    name="robot_state_subscriber",
)
```

CMake tracks the IDL as a build input instead of requiring developers to keep a
manually generated Python file up to date. For IDL files that include other IDL
files, add those files to `DEPENDS` and pass their directories to `rtiddsgen`
with `-I`.

Both endpoints must use compatible definitions of the same DDS type. The
example applications demonstrate this with two unrelated types to verify that
the operators are not coupled to a specific IDL.

## Configure QoS

The operators create their DDS entities using Connext's default QoS. They do
not hardcode reliability, durability, history, or a named profile.

Place a `USER_QOS_PROFILES.xml` file in the application's container working
directory and mark the desired profile with `is_default_qos="true"`. Connext
loads it automatically when the application starts. The supplied examples use
matching `RELIABLE`, `KEEP_ALL`, and `TRANSIENT_LOCAL` reader/writer settings so
all 20 samples of each type are retained while the writers exist.

Those settings are appropriate for a finite demonstration, not necessarily for
an unbounded stream. Production applications should select explicit resource
limits, durability depth, blocking behavior, and failure handling for their
data rates and lifecycle.

## Build and test

Build the operator environment through HoloHub:

```bash
./holohub build-container dds_pubsub
```

Run the Python operator tests in that container:

```bash
./holohub run-container dds_pubsub -- \
  "python3 -m pytest operators/dds/dds_pubsub/python/tests -v"
```

The reference workflow is container-only. A local installation must reproduce
the matching Holoscan SDK, Connext 7.7.0 runtime, Connext Python API, license,
and environment configuration and is outside this module's documented setup.

## Ownership and contact

Vendor: RTI Real-Time Innovations

Contact: `juanca@rti.com`

## Learn more

- [RTI Connext Professional](https://www.rti.com/products/connext-professional)
- [RTI and NVIDIA](https://www.rti.com/products/third-party-integrations/nvidia)
- [The Connext databus](https://www.rti.com/products/what-is-a-databus)
- [The DDS standard](https://www.rti.com/products/dds-standard)
