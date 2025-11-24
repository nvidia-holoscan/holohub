### DDS Base Operator

The DDS Base Operator provides a base class which can be inherited by any
operator class which requires access to a DDS domain.

This operator requires an installation of [RTI Connext](https://content.rti.com/l/983311/2025-07-08/q5x1n8) to provide access to the DDS domain, as specified by the [OMG Data-Distribution Service](https://www.omg.org/omg-dds-portal/). To obtain a license/activation key, please [click here](https://content.rti.com/l/983311/2025-07-25/q6729c). Additional information on Connext can be found here: https://www.rti.com/products/third-party-integrations/nvidia.  For any questions, please email evaluations@rti.com

#### `holoscan::ops::DDSOperatorBase`

Base class which provides the parameters and members required to access a
DDS domain.

For more documentation about how these parameters (and other similar
inheriting-class parameters) are used, see the
[RTI Connext Documentation](https://community.rti.com/documentation).

##### Parameters

- **`qos_provider`**: URI for the DDS QoS Provider
  - type: `std::string`
- **`participant_qos`**: Name of the QoS profile to use for the DDS DomainParticipant
  - type: `std::string`
- **`domain_id`**: The ID of the DDS domain to use
  - type: `uint32_t`
