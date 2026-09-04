# Connext DDS Subscriber

Before running this application, read the
[RTI Connext DDS Module overview](../../../modules/holoscan-connext-dds/README.md).

This Holoscan application waits for 20 `Telemetry` samples and 20 `Command`
samples from a separate `connext_dds_publisher` process. It validates every
identifier and exits automatically once both streams are complete.
