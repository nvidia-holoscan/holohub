### DDS Shape Subscriber Operator

The DDS Shape Subscriber Operator subscribes to and reads from the `Square`, `Circle`, and
`Triangle` shape topics as used by the [RTI Shapes Demo](https://www.rti.com/free-trial/shapes-demo).
It will then translate the received shape data to an internal `Shape` datatype for output
to downstream operators.

This operator requires an installation of [RTI Connext](https://content.rti.com/l/983311/2025-07-08/q5x1n8) to provide access to the DDS domain, as specified by the [OMG Data-Distribution Service](https://www.omg.org/omg-dds-portal/). To obtain a license/activation key, please [click here](https://content.rti.com/l/983311/2025-07-25/q6729c). Additional information on Connext can be found here: https://www.rti.com/products/third-party-integrations/nvidia.  For any questions, please email evaluations@rti.com

#### `holoscan::ops::DDSShapesSubscriberOp`

Operator class for the DDS Shapes Subscriber.

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

##### Parameters

- **`reader_qos`**: The name of the QoS profile to use for the DDS DataReader
  - type: `std::string`

##### Outputs

- **`output`**: Output shapes, translated from those read from DDS
  - type: `holoscan::ops::DDSShapesSubscriberOp::Shape`
