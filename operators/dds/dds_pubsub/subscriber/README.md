# Connext DDS Subscriber Operator

Before using this operator, read the
[RTI Connext DDS Module overview](../../../../modules/holoscan-connext-dds/README.md)
for supported versions, container requirements, and license setup.

`holohub.dds.ConnextDDSSubscriberOp` receives strongly typed RTI Connext DDS
samples and emits them into a Holoscan graph without changing their
application-defined Python type.

## Output

- **`output`**: one valid instance of the configured `topic_type`.

## Arguments

- **`domain_id`** (`int`, required): DDS domain used by the participant.
- **`topic`** (`str`, required): DDS topic name.
- **`topic_type`** (Python type, required): Connext-compatible type registered
  for the topic.

## Behavior and lifecycle

The operator creates its participant, topic, DataReader, and selector in
`start()`. The selector takes at most one data sample per compute call. When no
sample is available, the operator emits nothing and returns without blocking.
DDS entities are closed in `stop()`.

## Usage

```python
from rti import idl

from holohub.dds import ConnextDDSSubscriberOp


@idl.struct
class RobotState:
    sequence: int = 0
    position: float = 0.0


subscriber = ConnextDDSSubscriberOp(
    fragment,
    domain_id=42,
    topic="RobotState",
    topic_type=RobotState,
    name="robot_state_subscriber",
)
```

The next Holoscan operator receives the same DDS-compatible object and can
access its fields directly:

```python
sample = op_input.receive("input")
print(sample.sequence, sample.position)
```

## Generate the subscriber type from IDL

Keep the IDL and generated code with the application that owns the data
contract. Given `RobotState.idl`:

```idl
struct RobotState {
    int64 sample_id;
    double position;
};
```

add RTI Code Generator to the application CMake build:

```cmake
set(IDL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/RobotState.idl")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GENERATED_TYPE "${GENERATED_DIR}/RobotState.py")
set(RTIDDSGEN_HOME "${CMAKE_CURRENT_BINARY_DIR}/rti_home")

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
          "${RTIDDSGEN_EXECUTABLE}" -language python -replace
          -d "${GENERATED_DIR}" "${IDL_FILE}"
  DEPENDS "${IDL_FILE}"
  COMMENT "Generating Python DDS type from RobotState.idl"
  VERBATIM
)

add_custom_target(robot_state_subscriber ALL
  DEPENDS dds_pubsub_python "${GENERATED_TYPE}"
)
```

The module container provides `rtiddsgen` and `NDDSHOME`. Import the generated
module and pass `robot_state_idl.RobotState` as `topic_type`. Publisher and
subscriber must use compatible definitions. For included IDL files, add them
to `DEPENDS` and pass their directories with `-I`.

## QoS

The operator does not hardcode a named profile or individual QoS policies. Its
DataReader uses Connext's default QoS, including a default XML profile when one
is configured. Place `USER_QOS_PROFILES.xml` in the application's container
working directory and mark the intended profile with `is_default_qos="true"`.

The reader QoS must be compatible with the writer. Production applications
must choose reliability, durability, history, resource limits, and failure
handling for their data rate and lifecycle. The example's `RELIABLE`,
`KEEP_ALL`, and `TRANSIENT_LOCAL` settings are intended for a finite 20-sample
validation.

## Scheduling limitation

The operator is nonblocking, but the current Python implementation relies on
normal Holoscan scheduling rather than a DDS-driven scheduler notification.
It checks for one sample whenever scheduled and emits nothing if none is ready.

## Test

Run the publisher and subscriber unit tests through HoloHub:

```bash
./holohub test dds_pubsub --language python
```

The subscriber tests cover sample output, the no-data path, and DDS entity
cleanup.
