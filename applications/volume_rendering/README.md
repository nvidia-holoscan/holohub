# Volume rendering using ClaraViz

![](screenshot.png)<br>

This application loads a medical CT scan and renders it in real time at interactive frame rates using ClaraViz (https://github.com/NVIDIA/clara-viz).

The application uses the `VolumeLoaderOp` operator to load the medical volume data, the `VolumeRendererOp` operator to render the volume and the `HolovizOp` operator to display the result and handle the camera movement.

### Data

You can find CT scan datasets for use with this application from [embodi3d](https://www.embodi3d.com/).

#### Importing CT datasets

This section describes the steps to user CT datasets additionally to the dataset provided by the volume rendering application.

First get the data in a supported format. Supported formats are:
* [MHD](https://itk.org/Wiki/ITK/MetaIO/Documentation)
* [NIFTI](https://nifti.nimh.nih.gov/)
* [NRRD](https://teem.sourceforge.net/nrrd/format.html)

CT Data for the example dataset is available in the `data/volume_rendering` folder.

Additionally information on lighting, transfer functions and other settings is needed for the renderer to create an image. These settings are loaded from JSON files. The JSON files for the included example dataset is here `data/volume_rendering/config.json`.

There are two options to create a config file for a new dataset. First, use the example config as a reference to create a new config and modify parameters. Or let the renderer create a config file with settings deduced from the dataset.

Assuming the volume file is is named `new_volume.nrrd`. Specify the new volume file (`-d new_volume.nrrd`), set the config file option to an empty string (`-c ""`) to force the renderer to deduce settings and specify the name of the config file to write (`-w new_config.json`):

```bash
  ./applications/volume_rendering/volume_rendering -d new_volume.nrrd -c "" -w new_config.json
```

This will create a file `new_config.json`. If there is a segmentation volume present add it with `-m new_seg_volume.nrrd`.

By default the configuration is set up for rendering still images. For interactive rendering change the `timeSlot` setting in `RenderSettings` to the desired frame time in milliseconds, e.g. `33.0` for 30 fps.

Also by default all lights and the background are show in the scene. To avoid this change all `"show": true,` values to `"show": false,`.

Modify the configuration file to your needs. To display the volume with the new configuration file add the configuration with the `-c new_config.json` argument:

```bash
  ./applications/volume_rendering/volume_rendering -d new_volume.nrrd -c new_config.json
```

#### Transfer functions

Usually CT datasets are stored in [Hounsfield scale](https://en.wikipedia.org/wiki/Hounsfield_scale). The renderer maps these values in Hounsfiled scale to opacity in order to display the volume. These mappings are called transfer functions. Multiple transfer functions for different input value regions can be defined. Transfer functions also include material properties like diffuse, specular and emissive color. The range of input values the transfer function is applied to is in normalized input range `[0, 1]`.

#### Segmentation volume

Different organs often have very similar Hounsfield values, therefore additionally an segmentation volume is supported. The segmentation volume contains an integer index for each element of the volume. Transfer functions can be restricted on specific segmentation indices. The segmentation volume can, for example, be generated using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).

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

### Command Line Arguments

```
Holoscan ClaraViz volume renderer.
Usage: ./applications/volume_rendering/volume_rendering [options]
Options:
  -h, --help                            Display this information
  -c <FILENAME>, --config <FILENAME>    Name of the renderer JSON configuration file to load (default '../data/volume_rendering/config.json')
  -w <FILENAME>, --write_config <FILENAME> Name of the renderer JSON configuration file to write to (default '')
  -d <FILENAME>, --density <FILENAME>   Name of density volume file to load (default '../data/volume_rendering/highResCT.mhd')
  -m <FILENAME>, --mask <FILENAME>      Name of mask volume file to load (default '../data/volume_rendering/smoothmasks.seg.mhd')
  -n <COUNT>, --count <COUNT>           Duration to run application (default '-1' for unlimited duration)
```