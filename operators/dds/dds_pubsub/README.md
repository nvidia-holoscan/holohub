# DDS Publish and Subscribe Operators

> **Start here:** Read the
> [RTI Connext DDS Module overview](../../../modules/holoscan-connext-dds/README.md)
> first for the Connext introduction, supported versions, license, requirements,
> and container workflow.

`ConnextDDSPublisherOp` and `ConnextDDSSubscriberOp` connect Holoscan flows to
RTI Connext DDS. The operators are independent of a particular IDL: applications
provide the Python type generated from their IDL (or declared with `rti.idl`) as
the `topic_type` argument.

## Container build

Build the operator container with:

```bash
./holohub build-container dds_pubsub
```

The Dockerfile accepts `RTI_CONNEXT_VERSION` as a build argument, although the
operator metadata and default environment currently target the version listed
in the module overview.

## Usage

```python
import rti.idl as idl

from holohub.dds import ConnextDDSPublisherOp, ConnextDDSSubscriberOp


@idl.struct
class RobotState:
    position: float = 0.0


publisher = ConnextDDSPublisherOp(
    fragment,
    domain_id=9,
    topic="RobotState",
    topic_type=RobotState,
    name="dds_publisher",
)

subscriber = ConnextDDSSubscriberOp(
    fragment,
    domain_id=9,
    topic="RobotState",
    topic_type=RobotState,
    name="dds_subscriber",
)
```

The publisher has an `input` port and validates that each incoming sample is an
instance of `topic_type`. The subscriber has an `output` port.

Both operators use Connext's default QoS without overriding policies in Python.
Applications can provide a `USER_QOS_PROFILES.xml` in the container's working
directory, with a profile marked `is_default_qos="true"`, before starting the
process. Connext loads it automatically. Without XML or other default QoS
configuration, Connext's built-in defaults apply; the operators do not impose
reliability, durability, or history settings.

The example supplies an application-owned XML profile with matching
`RELIABLE`, `TRANSIENT_LOCAL`, and `KEEP_ALL` settings for its two topic types.
It leaves sample resource limits unlimited by default. For continuous streams,
review memory limits and historical retention. See the example's
[QoS scope](../../../applications/dds/README.md#qos-scope)
for limitations and XML configuration instructions.

IDL files and generated types belong to the applications or workflows that own
those data contracts; they are intentionally not bundled with these generic
operators.
