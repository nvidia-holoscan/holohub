# DDS Shape Subscriber Operator

Before using this operator, read the
[RTI Connext DDS Module overview](../../../modules/holoscan-connext-dds/README.md)
for the supported versions, container requirements, and license setup.

The DDS Shape Subscriber Operator subscribes to the `Square`, `Circle`, and
`Triangle` topics used by the
[RTI Shapes Demo](https://www.rti.com/free-trial/shapes-demo). It converts valid
`ShapeTypeExtended` samples into a representation suitable for downstream
Holoscan visualization operators.

## `holoscan::ops::DDSShapesSubscriberOp`

Operator class for the DDS Shapes Subscriber.

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

### Parameters

- **`reader_qos`** (`std::string`, default: empty): DataReader QoS profile name
  resolved by the inherited QoS provider. The same profile is used for all
  three readers.
- The operator also inherits `qos_provider`, `participant_qos`, and `domain_id`
  from [`DDSOperatorBase`](../base/README.md).

### Outputs

- **`output`** (`std::vector<holoscan::ops::DDSShapesSubscriberOp::Shape>`): all
  valid Square, Circle, and Triangle samples taken during the compute call.

## Conversion behavior

Each output shape contains its shape type, RGBA color, normalized position,
width, and height. Coordinates are normalized against the RTI Shapes Demo
publisher area of 235 by 265 pixels. The recognized color names are `PURPLE`,
`BLUE`, `RED`, `GREEN`, `YELLOW`, `CYAN`, `MAGENTA`, and `ORANGE`; unknown
names map to black.

## Limitations

- Topic names and the `ShapeTypeExtended` DDS type are fixed for compatibility
  with RTI Shapes Demo.
- The coordinate normalization assumes the 235-by-265 publisher area.
- Fill styles and rotation are not represented.
- The operator takes all currently available valid samples on each compute
  call and emits one vector, including an empty vector when none are available.
