# Object Detection using PyTorch Faster R-CNN

This application performs object detection using frcnn resnet50 model from torchvision.
The inference is executed using `torch` backend in `holoinfer` module in Holoscan SDK.

`object_detection_torch.yaml` is the configuration file. Input video file is converted into GXF tensors and the name and location of the GXF tensors are updated in the `basename` and the `directory` field in `replayer`.

This application need `Libtorch` for inferencing. Ensure that the Holoscan SDK is build with `build_libtorch` flag as true. If not, then rebuild the SDK with following: `./holohub build --build_libtorch true` before running this application.

## Data

To run this application, you will need the following:

- Model name: frcnn_resnet50_t.pt
    - The model should be converted to torchscript format.  The original pytorch model can be downloaded from [pytorch model](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html). `frcnn_resnet50_t.pt` is used
- Model configuration file: frcnn_resnet50_t.yaml
    - Model config documents input and output nodes, their dimensions and respective datatype.
- Labels file: labels.txt
    - Labels for identified objects.
- Postprocessor configuration file: postprocessing.yaml
    - This configuration stores the number and type of objects to be identified. By default, the application detects and generates bounding boxes for `car` (max 50), `person` (max 50), `motorcycle` (max 10) in the input frame. All remaining identified objects are tagged with label `object` (max 50).
    - Additionally, color of the bounding box for each identified object can be set.
    - Threshold of scores can be set in the `params`. Default value is 0.75.

Sample dataset can be any video file freely available for testing on the web. E.g. [Traffic video](https://www.pexels.com/video/cars-on-highway-854671/)

Once the video is downloaded, it must be [converted into GXF entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#usage). As shown in the command below, width and height is set to 1920x1080 by default. To reduce the size of generated tensors a lower resolution can be used. Generated entities must be saved at <data_dir>/object_detection_torch folder.

```bash
ffmpeg -i <downloaded_video> -pix_fmt rgb24 -f rawvideo pipe:1 | python utilities/convert_video_to_gxf_entities.py --width 1920 --height 1080 --channels 3 --framerate 30
```

If resolution is updated in entity generation, it must be updated in the following config files as well:
<data_dir>/object_detection_torch/frcnn_resnet50_t.yaml
<data_dir>/object_detection_torch/postprocessing.yaml

## Quick start
If you want to quickly run this application, you can use the `./holohub run` command.

```sh
./holohub run object_detection_torch
```

Otherwise, you can build and run the application using the commands below.

## Building the application

The best way to run this application is inside the container, as it would provide all the required third-party packages:

```bash
# Create and launch the container image for this application
./holohub run-container object_detection_torch
# Build the application inside the container. Note that this downloads the video data as well
./holohub build object_detection_torch
# Generate the pytorch model
python3 applications/object_detection_torch/generate_resnet_model.py  data/object_detection_torch/frcnn_resnet50_t.pt
# Run the application
./holohub run object_detection_torch
```

Please refer to the top level Holohub README.md file for more information on how to build this application.

## Running the application

```bash
# ensure the current working directory contains the <data_dir>.
<build_dir>/object_detection_torch
```

If application is executed from within the holoscan sdk container and is not able to find `libtorch.so`, update `LD_LIBRARY_PATH` as below:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libtorch/1.13.1/lib
```

On aarch64, if application is executed from within the holoscan sdk container and libtorch throws linking errors, update the `LD_LIBRARY_PATH` as below:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
```

## Containerize the application

To containerize the application using [Holoscan CLI](https://docs.nvidia.com/holoscan/sdk-user-guide/cli/cli.html), first build the application using `./holohub install object_detection_torch`, run the `package-app.sh` script and then follow the generated output to package and run the application.

Refer to the [Packaging Holoscan Applications](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) section of the [Holoscan User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/) to learn more about installing the Holoscan CLI or packaging your application using Holoscan CLI.

## Known Issues

**Limited platform support**: The Faster R-CNN PyTorch model in this example relies on symbols available only in `torchvision<=0.23.0`. arm64 SBSA platforms and x86_64 platforms with CUDA Toolkit >= 13.0 are currently not supported as no compatible combination of Holoscan SDK, PyTorch (CUDA), and torchvision (CUDA) are available for these platforms.

At the time of writing, the supported configuration based on PyTorch components availability is as follows:
1. x86_64 with CUDA Toolkit <=12.9
  - Latest Holoscan SDK (>=v3.9.0)
  - PyTorch 2.8.0 with CUDA 12.9 support
  - torchvision 0.23.0 with CUDA 12.9 support
2. NVIDIA Orin platform running in integrated GPU mode (iGPU)
  - Latest Holoscan SDK (>=v3.9.0)
  - PyTorch 2.8.0 with CUDA 12.9 support
  - torchvision 0.23.0 with CUDA 12.9 support

The application container will fail to build on other platforms with the error:
```bash
ERROR: No matching distribution found
```