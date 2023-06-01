### QCAP

The `qcap_source` extension supports YUAN High-Tech capture cards as
the video source.

#### `yuan::holoscan::QCAPSource`

QCAP Source codelet

##### Parameters

- **`channel`**: Channel to use (default: `0`)
  - type: `uint32_t`
- **`width`**: Width of the stream (default: `3840`)
  - type: `uint32_t`
- **`height`**: Height of the stream (default: `2160`)
  - type: `uint32_t`
- **`framerate`**: Framerate of the stream (default: `60`)
  - type: `uint32_t`
- **`rdma`**: Enable RDMA (default: `false`)
  - type: `bool`
