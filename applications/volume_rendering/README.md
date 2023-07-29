# Volume rendering using ClaraViz

![](screenshot.png)<br>

This application loads a medical CT scan and renders it in real time at interactive frame rates using ClaraViz (https://github.com/NVIDIA/clara-viz).

The application uses the `VolumeLoaderOp` operator to load the medical volume data, the `VolumeRendererOp` operator to render the volume and the `HolovizOp` operator to display the result and handle the camera movement.

### Data

You can find CT scan datasets for use with this application from [embodi3d](https://www.embodi3d.com/).

## Run Instructions

From the build directory, run the command:

```bash
./applications/volume_rendering/volume_rendering
```

The path of the volume configuration file, volume density file and volume mask file can be passed to the application. Use

```bash
./applications/volume_rendering/volume_rendering -h
```

to get more information on command line parameters.
