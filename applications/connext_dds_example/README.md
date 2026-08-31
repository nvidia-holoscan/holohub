# Connext DDS example

This application validates the generic `connext_dds` operators with a DDS type
owned by the application rather than by the operators. It declares two
different types (`Telemetry` and `Command`) with `rti.idl`, publishes 20 samples
on separate topics, receives all 20 with the same generic operator classes, and
fails if either subscriber receives fewer than 20 samples. The example's XML
profile configures matching `RELIABLE`/`TRANSIENT_LOCAL`/`KEEP_ALL` QoS to retain
samples written before endpoint discovery completes while the writers exist.

## QoS scope

This is a short, finite demonstration that publishes 20 samples per topic.
The application configures `RELIABLE`, `KEEP_ALL`, and `TRANSIENT_LOCAL` in
[USER_QOS_PROFILES.xml](USER_QOS_PROFILES.xml) and leaves resource limits at
their defaults. These settings are intended to demonstrate receiving every
sample, not to prescribe a memory
budget for a long-running application.

With the default unlimited sample limits and automatic durability writer
depth, the publisher can retain an ever-growing history for late-joining
readers, including samples already acknowledged by existing readers.
`TRANSIENT_LOCAL` retains that history only while the writer exists; it is
not persistent storage. Before adapting this example to a continuous stream,
choose resource limits, durability writer depth, and blocking timeouts for
the workload, and handle backpressure and timeout errors explicitly.

### XML configuration

The operators do not hardcode QoS policies or profile names. They create DDS
entities using Connext's default QoS. Connext automatically loads
`USER_QOS_PROFILES.xml` from the process's current working directory and uses
the profile marked `is_default_qos="true"`. The example selects
`ConnextDDSExample::ReliableKeepAll` this way for both topic types.

CMake copies the XML into the application's build directory, which is the
working directory selected by `./holohub run connext_dds_example`. It also
installs the XML alongside the example script. No extra Docker mount or
Python configuration is needed for the supplied profile.

To experiment with different policies, edit the application's XML and rerun
the HoloHub command below, which rebuilds the application. No operator changes
are needed. For your own application, supply a `USER_QOS_PROFILES.xml` in its
container working directory **before starting the process**, with a default
profile and compatible reader/writer QoS. The automatic filename is
`USER_QOS_PROFILES.xml`, not `QoS.xml`. If no XML or other default QoS
configuration is supplied, Connext uses its built-in defaults; the operators
no longer impose reliable delivery or retention for late joiners. Changing
the example's policies can therefore cause its 20-sample check to fail.

See RTI's [default XML profile behavior](https://community.rti.com/static/documentation/connext-dds/current/doc/manuals/connext_dds_professional/code_generator/users_manual/code_generator/users_manual/GeneratingCode.htm)
for details on automatic loading and `is_default_qos`.

See RTI's [durability documentation](https://community.rti.com/static/documentation/connext-dds/current/doc/manuals/connext_dds_professional/users_manual/users_manual/DURABILITY_QosPolicy.htm)
for the relationship between history, writer depth, and resource limits.

## Build and run

Build it through the HoloHub container workflow:

```bash
./holohub build connext_dds_example
```

The complete example is intended to run in Docker through HoloHub so that no
Holoscan or RTI packages are installed into the host (especially an IGX
development machine):

Connext requires a valid license at runtime. Request and download the RTI
Connext activation key from
<https://content.rti.com/l/983311/2025-07-25/q6729c>, then save it as
`rti_license.dat` in the HoloHub repository root. Run the application through
HoloHub and mount the license at the path configured by the container:

```bash
curl -L https://content.rti.com/l/983311/2025-07-25/q6729c -o rti_license.dat
./holohub run connext_dds_example \
  --docker-opts="-v $(pwd)/rti_license.dat:/opt/rti.com/rti_connext_dds-7.7.0/rti_license.dat:ro"
```

The same container workflow can be used on other systems; a local run would
need to reproduce all image dependencies and is left as an advanced adaptation.

No generated DDS schema is stored in the operator. A production application
can replace either inline declaration with a Python type generated from its own
IDL file without changing either operator. For example, generate Python types
with RTI's `rtiddsgen` inside the module container, then pass the generated
class as `topic_type`:

```python
from my_generated_idl import RobotState

publisher = ConnextDDSPublisherOp(
    fragment,
    domain_id=42,
    topic="RobotState",
    topic_type=RobotState,
    name="robot_state_publisher",
)
```

The publisher and subscriber must use the same topic name, domain, and generated
type (and compatible QoS) on both sides. Keep the IDL and generated files in the
application or workflow that owns that data contract; the generic module does
not bundle application-specific schemas.
