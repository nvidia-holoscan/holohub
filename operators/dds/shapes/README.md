### DDS Shape Subscriber Operator

The DDS Shape Subscriber Operator subscribes to and reads from the `Square`, `Circle`, and
`Triangle` shape topics as used by the [RTI Shapes Demo](https://www.rti.com/free-trial/shapes-demo).
It will then translate the received shape data to an internal `Shape` datatype for output
to downstream operators.

#### `holoscan::ops::DDSShapesSubscriberOp`

Operator class for the DDS Shapes Subscriber.

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

##### Parameters

- **`reader_qos`**: The name of the QoS profile to use for the DDS DataReader
  - type: `std::string`

##### Outputs

- **`output`**: Output shapes, translated from those read from DDS
  - type: `holoscan::ops::DDSShapesSubscriberOp::Shape`
