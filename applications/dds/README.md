# DDS applications

> **Start here:** Read the
> [RTI Connext DDS Module overview](../../modules/holoscan-connext-dds/README.md)
> first for the Connext introduction, supported versions, license, requirements,
> and container workflow.

This directory contains the existing `dds_video` application and three
applications that exercise the generic `dds_pubsub` operators with two
application-owned DDS types: `Telemetry` and `Command`.

- `connext_dds_roundtrip` is a self-contained smoke test. It publishes and
  receives both streams in one Holoscan process.
- `connext_dds_publisher` publishes 20 samples of each type and then exits.
- `connext_dds_subscriber` runs in a separate Holoscan process, validates all
  40 identifiers, and exits when both streams are complete.

The three Connext examples share their IDL declarations, graph components,
constants, and XML QoS configuration from `connext_dds_common/`; the publisher,
subscriber, and round-trip entrypoints only compose the components they need.

## QoS scope

These are short, finite demonstrations that publish 20 samples per topic. The
shared [USER_QOS_PROFILES.xml](connext_dds_common/USER_QOS_PROFILES.xml) configures
`RELIABLE`, `KEEP_ALL`, and `TRANSIENT_LOCAL` QoS and leaves resource limits at
their defaults. These settings demonstrate receiving every sample; they do not
prescribe a memory budget for a long-running application.

With unlimited sample limits and automatic durability writer depth, a
publisher can retain a growing history for late-joining readers, including
samples already acknowledged by existing readers. `TRANSIENT_LOCAL` retains
that history only while the writer exists; it is not persistent storage. Before
adapting the example to a continuous stream, select resource limits, durability
writer depth, and blocking timeouts for the workload, and handle backpressure
and timeout errors explicitly.

The operators do not hardcode QoS policies or profile names. Connext
automatically loads `USER_QOS_PROFILES.xml` from the process working directory
and uses the profile marked `is_default_qos="true"`. Each example's CMake
configuration copies the shared XML into its own build directory before it is
run.

For another application, provide a `USER_QOS_PROFILES.xml` in its container
working directory before starting the process. Without XML or another default
QoS configuration, Connext uses its built-in defaults.

## Container-only setup

Complete the module overview's container and license setup first. The commands
below expect `rti_license.dat` in the HoloHub repository root and mount it
read-only into the Connext runtime. HoloHub selects the module Dockerfile from
application metadata; do not pass `--local` or `--docker-file`.

## Run the round-trip smoke test

Build and run the self-contained application:

```bash
./holohub build connext_dds_roundtrip --language python
./holohub run connext_dds_roundtrip --language python \
  --docker-opts="-v ${PWD}/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

A successful run ends with:

```text
DDS round trip succeeded: Telemetry=20, Command=20 samples
```

## Run two independent Holoscan applications

Open two terminals in the HoloHub repository root. Start the subscriber first
so its DDS readers are available while the transient-local writers exist.

Terminal 1:

```bash
./holohub run connext_dds_subscriber --language python \
  --docker-opts="-v ${PWD}/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

The subscriber waits until it has received both complete streams. In terminal
2, start the independent publisher:

```bash
./holohub run connext_dds_publisher --language python \
  --docker-opts="-v ${PWD}/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

The publisher and subscriber run in different Docker containers and Holoscan
processes. HoloHub gives both containers host networking, allowing Connext DDS
discovery and data exchange. Their final messages should be:

```text
DDS publisher succeeded: Telemetry=20, Command=20 samples
DDS subscriber succeeded: Telemetry=20, Command=20 samples
```

If the publisher is started first and exits before the subscriber discovers
it, its transient-local history no longer exists. Start the subscriber first
for this finite demonstration.

## Use another IDL

No generated DDS schema is stored in the generic operator. A production
application can replace the inline example declarations with Python types
generated from its own IDL without changing either operator. The
[module documentation](../../modules/holoscan-connext-dds/README.md#use-your-own-idl)
shows how to make the IDL part of the application's CMake build so the Python
type is regenerated and installed automatically. Pass that generated class as
`topic_type`:

```python
from my_generated_idl import RobotState

from holohub.dds import ConnextDDSPublisherOp

publisher = ConnextDDSPublisherOp(
    fragment,
    domain_id=42,
    topic="RobotState",
    topic_type=RobotState,
    name="robot_state_publisher",
)
```

The communicating applications must use the same domain ID, topic name, DDS
type, and compatible QoS. Keep the IDL and generated files with the application
or workflow that owns that data contract.
