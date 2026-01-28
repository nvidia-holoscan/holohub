# VTK Renderer Operator

The `vtk_renderer` extension takes the output of the source video player and the
output of the `tool_tracking_postprocessor` operator and renders the video
stream with an overlay annotation of the label using VTK.

VTK can be a useful addition to holohub stack since VTK is a industry leading
visualization toolkit. It is important to mention that this renderer operator
needs to copy the input from device memory to host due to limitations of VTK.
While this is a strong limitation for VTK we believe that VTK can still be a
good addition and VTK is an evolving project. Perhaps in the future we could
overcome this limitation.

## How to build this operator

Using Holohub CLI, you can create and run a container, which includes VTK,
by running the following command from the root directory of Holohub:

```bash
./holohub run-container vtk_renderer
```

This command will create  and run a container based on the provided [`Dockerfile`](./Dockerfile).

> [!NOTE]
> If you want to only build the docker image without running it, you can use the following command,
> which will create the image and tag it as `holohub:vtk_renderer`.
>
> ```bash
> ./holohub build-container vtk_renderer
> ```

Inside the container you can build the holohub application with:

```bash
./holohub build <application> --build-with vtk_renderer
```

## Parameters

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
