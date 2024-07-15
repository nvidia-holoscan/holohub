### vtk_renderer operator

The `vtk_renderer` extension takes the output of the source video player and the
output of the `tool_tracking_postprocessor` operator and renders the video
stream with an overlay annotation of the label using VTK.

VTK can be a useful addition to holohub stack since VTK is a industry leading
visualization toolkit. It is important to mention that this renderer operator
needs to copy the input from device memory to host due to limitations of VTK.
While this is a strong limitation for VTK we believe that VTK can still be a
good addition and VTK is an evolving project. Perhaps in the future we could
overcome this limitation.

#### How to build this operator

Build the HoloHub container as described at the root [README.md](../../README.md)

You need to create a docker image which includes VTK with the provided
`vtk.Dockerfile`:

```bash
docker build -t vtk:latest -f vtk.Dockerfile .
```

Then, you can build the tool tracking application with the provided
`Dockerfile`:

```bash
./dev_container launch --img vtk:latest
```

Inside the container you can build the holohub application with:

```bash
./run build <application> --with vtk_renderer
```

##### Parameters

- **`videostream`**: Input channel for the videostream, type `gxf::Tensor`
  - type: `gxf::Handle<gxf::Receiver>`
- **`annotations`**: Input channel for the annotations, type `gxf::Tensor`
  - type: `gxf::Handle<gxf::Receiver>`
- **`window_name`**: Compositor window name.
  - type: `std::string`
- **`width`**: width of the renderer window.
  - type: `int`
- **`height`**: height of the renderer window.
  - type: `int`
- **`labels`**: labels to be displayed on the rendered image.
  - type: `std::vector<std::string>>`
