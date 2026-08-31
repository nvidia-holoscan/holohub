# RTI Connext DDS Operators

`ConnextDDSPublisherOp` and `ConnextDDSSubscriberOp` connect Holoscan flows to
RTI Connext DDS. The operators are independent of a particular IDL: applications
provide the Python type generated from their IDL (or declared with `rti.idl`) as
the `topic_type` argument.

## Requirements

- RTI Connext DDS **7.7.0** and its matching Python API (`rti.connext==7.7.0`)
- A valid RTI Connext activation key available through `RTI_LICENSE_FILE` at runtime
- Matching topic names, domain IDs, types, and QoS between communicating peers

The provided [Dockerfile](Dockerfile) installs both the Connext SDK from RTI's
official APT repository and the Python API. The license is proprietary and is
not distributed with HoloHub.

## RTI license

1. [Request and download an RTI Connext activation key](https://content.rti.com/l/983311/2025-07-25/q6729c).
2. Alternatively, download the provided evaluation license directly:

   ```bash
   curl -L https://content.rti.com/l/983311/2025-07-25/q6729c -o rti_license.dat
   ```

3. Save the file as `rti_license.dat` in the HoloHub repository root.
   This filename is ignored by Git and must not be committed.
4. Mount it into the container when it is launched:

   ```bash
   ./holohub run-container connext_dds \
     --docker-opts="-v $(pwd)/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
   ```

The container sets `RTI_LICENSE_FILE` to that destination. For a local
installation, set it explicitly instead:

```bash
export RTI_LICENSE_FILE=/absolute/path/to/rti_license.dat
```

See the [RTI Connext Express usage rules](https://www.rti.com/products/connext-express)
for the applicable license terms.

## Container build

Build the operator container with:

```bash
./holohub build-container connext_dds
```

The Dockerfile accepts `RTI_CONNEXT_VERSION` as a build argument, although the
operator metadata and default environment currently target Connext 7.7.0.

## Usage

```python
import rti.idl as idl

from holohub.connext_dds import ConnextDDSPublisherOp, ConnextDDSSubscriberOp


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
instance of `topic_type`. The subscriber has an `output` port and uses
`RELIABLE` reliability, `TRANSIENT_LOCAL` durability, and `KEEP_ALL` history.
The publisher uses matching QoS so samples are retained while late discovery
completes and are not intentionally discarded by a one-sample history policy.

IDL files and generated types belong to the applications or workflows that own
those data contracts; they are intentionally not bundled with these generic
operators.
