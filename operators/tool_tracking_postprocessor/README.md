# Tool Tracking Postprocessor Operator

The `tool_tracking_postprocessor` extension provides a codelet that converts inference output of `lstm_tensor_rt_inference` used in the endoscopy tool tracking pipeline to be consumed by the `holoviz` codelet.

#### `nvidia::holoscan::tool_tracking_postprocessor`

Tool tracking postprocessor codelet

##### Parameters

- **`in`**: Input channel, type `gxf::Tensor`
  - type: `gxf::Handle<gxf::Receiver>`
- **`out`**: Output channel, type `gxf::Tensor`
  - type: `gxf::Handle<gxf::Transmitter>`
- **`min_prob`**: Minimum probability, (default: 0.5)
  - type: `float`
- **`overlay_img_colors`**: Color of the image overlays, a list of RGB values with components between 0 and 1, (default: 12 qualitative classes color scheme from colorbrewer2)
  - type: `std::vector<std::vector<float>>`
- **`device_allocator`**: Output Allocator
  - type: `gxf::Handle<gxf::Allocator>`
- **`cuda_stream_pool`**: Instance of gxf::CudaStreamPool
  - type: `gxf::Handle<gxf::CudaStreamPool>`
