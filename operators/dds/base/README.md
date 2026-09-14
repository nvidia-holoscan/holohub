# DDS Base Operator

Before using this operator, read the
[RTI Connext DDS Module overview](../../../modules/holoscan-connext-dds/README.md)
for the supported versions, container requirements, and license setup.

The DDS Base Operator provides the common RTI Connext participant and QoS
setup used by the C++ DDS operators. HoloHub builds it as an internal
dependency; it is not a standalone graph operator.

## `holoscan::ops::DDSOperatorBase`

This class initializes a `dds::core::QosProvider` and a
`dds::domain::DomainParticipant`, then exposes them to derived operators.
Operators configured with the same QoS provider URI, participant QoS profile,
and domain ID share a participant while they are alive. A different value in
any of those fields creates a separate participant.

For more documentation about how these parameters (and other similar
inheriting-class parameters) are used, see the
[RTI Connext Documentation](https://community.rti.com/documentation).

### Parameters

- **`qos_provider`** (`std::string`, default: `qos_profiles.xml`): URI passed to
  the Connext `QosProvider`.
- **`participant_qos`** (`std::string`, default: empty): participant QoS profile
  name. An empty name asks the provider for its default participant QoS.
- **`domain_id`** (`uint32_t`, default: `0`): DDS domain used by the participant.

## Usage by derived operators

Derived operators must call `DDSOperatorBase::setup()` and
`DDSOperatorBase::initialize()` before using the protected `qos_provider_` and
`participant_` members. The video and Shapes operators in this module provide
working examples.

Entity-specific profiles such as `reader_qos` and `writer_qos` belong to the
derived operator. The base class only owns participant-level configuration.
