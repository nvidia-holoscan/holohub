# Connext DDS example

This application validates the generic `connext_dds` operators with a DDS type
owned by the application rather than by the operators. It declares two
different types (`Telemetry` and `Command`) with `rti.idl`, publishes 20 samples
on separate topics, receives all 20 with the same generic operator classes, and
fails if either subscriber receives fewer than 20 samples. Both operators use
matching `RELIABLE`/`TRANSIENT_LOCAL`/`KEEP_ALL` QoS, including samples written
before endpoint discovery completes.

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
