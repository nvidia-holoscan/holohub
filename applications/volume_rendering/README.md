# Volume rendering using ClaraViz

![](screenshot.png)<br>

This application loads a medical CT scan and renders it in real time at interactive frame rates using ClaraViz (https://github.com/NVIDIA/clara-viz).

The application uses the `VolumeLoaderOp` operator to load the medical volume data, the `VolumeRendererOp` operator to render the volume and the `HolovizOp` operator to display the result and handle the camera movement.

### Data

You can find CT scan datasets for use with this application from [embodi3d](https://www.embodi3d.com/).

Datasets are bundled with a default ClaraViz JSON configuration file for volume rendering. See [`VolumeRendererOp` documentation](/operators/volume_renderer/README.md#configuration) for details on configuration schema.

See [`VolumeLoaderOp` documentation](/operators/volume_loader/README.md#supported-formats) for supported volume formats.

## Build and Run Instructions

To build and run this application, use the ```dev_container``` script:

```bash
# C++
 ./dev_container build_and_run volume_rendering --language cpp

 # Python
  ./dev_container build_and_run volume_rendering --language python
```

The path of the volume configuration file, volume density file and volume mask file can be passed to the application.

You can use the following command to get more information on command line parameters for this application:

```bash
./dev_container build_and_run volume_rendering --language [cpp|python] --run_args --usages
```
