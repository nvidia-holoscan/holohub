### vtk_renderer operator

The `vtk_renderer` extension takes the output the source video player and the
output of the `tool_tracking_postprocessor` operator and renders the video
stream with an overlay annotation of the label of the tool being used.

##### Parameters

- **`videostream`**: Input channel for the videostream, type `gxf::Tensor`
  - type: `gxf::Handle<gxf::Receiver>`
- **`annotations`**: Input channel for the annotations, type `gxf::Tensor`
  - type: `gxf::Handle<gxf::Receiver>`
- **`width`**: width of the renderer window.
  - type: `int`
- **`height`**: height of the renderer window.
  - type: `int`
- **`labels`**: labels to be displayed on the rendered image.
  - type: `int`
