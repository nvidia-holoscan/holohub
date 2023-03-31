### Emergent

The `emergent_source` extension supports an Emergent Vision Technologies camera as
the video source. The datastream from this camera is transferred through Mellanox
ConnectX SmartNIC using Rivermax SDK.

#### `nvidia::holoscan::EmergentSource`

Emergent Source codelet

##### Parameters

- **`signal`**: Output signal
  - type: `gxf::Handle<gxf::Transmitter>`
- **`width`**: Width of the stream (default: `4200`)
  - type: `uint32_t`
- **`height`**: Height of the stream (default: `2160`)
  - type: `uint32_t`
- **`framerate`**: Framerate of the stream (default: `240`)
  - type: `uint32_t`
- **`rdma`**: Enable RDMA (default: `false`)
  - type: `bool`
