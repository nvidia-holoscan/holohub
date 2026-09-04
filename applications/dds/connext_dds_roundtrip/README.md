# Connext DDS Round Trip

Before running this application, read the
[RTI Connext DDS Module overview](../../../modules/holoscan-connext-dds/README.md).

This smoke-test application publishes and receives both example DDS streams in
one Holoscan process. It validates all identifiers and exits with an error if
any of the 40 expected samples is missing.
