### OpenIGTLink operator

The `openigtlink` operator provides a way to send and receive imaging data using the [OpenIGTLink](http://openigtlink.org/) library. The `openigtlink` operator contains separate operators for transmit and receive. Users may choose one or the other, or use both in applications requiring bidirectional traffic.

The `openigtlink` operators use class names: `OpenIGTLinkTxOp` and `OpenIGTLinkRxOp`

#### `nvidia::holoscan::openigtlink`

Operator class to send and transmit data using the OpenIGTLink protocol.

##### Receiver Configuration Parameters

- **`port`**: Port number of server
  - type: `integer`
- **`out_tensor_name`**: Name of output tensor
  - type: `string`
- **`flip_width_height`**: Flip width and height (necessary for receiving from 3D Slicer)
  - type: `bool`

##### Transmitter Configuration Parameters

- **`device_name`**: OpenIGTLink device name
  - type: `string`
- **`input_names`**: Names of input messages
  - type: `std::vector<std::string>`
- **`host_name`**: Host name
  - type: `string`
- **`port`**: Port number of server
  - type: `integer`