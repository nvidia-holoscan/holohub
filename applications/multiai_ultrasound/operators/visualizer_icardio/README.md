### Visualizer iCardio

The `visualizer_icardio` extension generates the visualization components from the processed results of the plax chamber model.

#### `nvidia::holoscan::multiai::VisualizerICardio`

Visualizer iCardio extension ingests the processed results of the plax chamber model and generates the key points, the key areas and the lines that are transmitted to the HoloViz codelet.

##### Parameters

- **`in_tensor_names_`**: Input tensor names
  - type: `std::vector<std::string>`
- **`out_tensor_names_`**: Output tensor names
  - type: `std::vector<std::string>`
- **`allocator_`**: Memory allocator
  - type: `gxf::Handle<gxf::Allocator>`
- **`receivers_`**: Vector of input receivers. Multiple receivers supported.
  - type: `HoloInfer::GXFReceivers`
- **`transmitter_`**: Output transmitter. Single transmitter supported.
  - type: `HoloInfer::GXFTransmitters`
