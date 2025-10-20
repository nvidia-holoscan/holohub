# Endoscopy Tool Segmentation from MONAI Model Zoo

This endoscopy tool segmentation application runs the MONAI Endoscopic Tool Segmentation from [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/endoscopic_tool_segmentation).

This HoloHub application has been verified on the GI Genius sandbox and is currently deployable to GI Genius Intelligent Endoscopy Modules. [GI Genius](https://www.cosmoimd.com/gi-genius/) is Cosmo Intelligent Medical Devices’ AI-powered endoscopy system. This implementation by Cosmo Intelligent Medical Devices showcases the fast and seamless deployment of HoloHub applications on products/platforms running on NVIDIA Holoscan.

## Model
We will be deploying the endoscopic tool segmentation model from [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/endoscopic_tool_segmentation). <br>
Note that you could also use the MONAI model zoo repo for training your own semantic segmentation model with your own data, but here we are directly deploying the downloaded MONAI model checkpoint into Holoscan. 

You can choose to 
- download the [MONAI Endoscopic Tool Segmentation Model on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/monai_endoscopic_tool_segmentation_model) directly and skip the rest of this Model section, or 
- go through the following conversion steps yourself.

### Model conversion to ONNX (optional)
Before deploying the MONAI Model Zoo's trained model checkpoint in Holoscan SDK, we convert the model checkpoint into ONNX. <br>

 1. Download the PyTorch model checkpoint linked in the README of [endoscopic tool segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/endoscopic_tool_segmentation#model-overview). We will assume its name to be `model.pt`.
 2. Clone the MONAI Model Zoo repo. 
```
cd [your-workspace]
git clone https://github.com/Project-MONAI/model-zoo.git
```
and place the downloaded PyTorch model into `model-zoo/models/endoscopic_tool_segmentation/`.

 3. Pull and run the docker image for [MONAI](https://hub.docker.com/r/projectmonai/monai). We will use this docker image for converting the PyTorch model to ONNX. 
```
docker pull projectmonai/monai
docker run -it --rm --gpus all -v [your-workspace]/model-zoo:/workspace/model-zoo -w /workspace/model-zoo/models/endoscopic_tool_segmentation/ projectmonai/monai
```
 4. Install onnxruntime within the container
 ```
pip install onnxruntime onnx-graphsurgeon
 ```
 5. Convert model
 
We will first export the model.pt file to ONNX by using the [export_to_onnx.py](https://github.com/Project-MONAI/model-zoo/blob/dev/models/endoscopic_tool_segmentation/scripts/export_to_onnx.py) file. Modify the backbone in [line 122](https://github.com/Project-MONAI/model-zoo/blob/dev/models/endoscopic_tool_segmentation/scripts/export_to_onnx.py#L122) to be efficientnet-b2:
```
model = load_model_and_export(modelname, outname, out_channels, height, width, multigpu, backbone="efficientnet-b2")
```
Note that the model in the Model Zoo here was trained to have only two output channels: label 1 = tools, label 0 = everything else, but the same Model Zoo repo can be repurposed to train a model with a different dataset that has more than two classes.
```
python scripts/export_to_onnx.py --model model.pt --outpath model_endoscopic_tool_seg.onnx --width 736 --height 480 --out_channels 2
```
Fold constants in the ONNX model.
```
polygraphy surgeon sanitize --fold-constants model_endoscopic_tool_seg.onnx -o model_endoscopic_tool_seg_sanitized.onnx
```
Finally, modify the input and output channels to have shape [n, height, width, channels], [n, channels, height, width]. 
```
python scripts/graph_surgeon_tool_seg.py --orig_model model_endoscopic_tool_seg_sanitized.onnx --new_model model_endoscopic_tool_seg_sanitized_nhwc_in_nchw_out.onnx
```

## Data
For this application we will use the same [Endoscopy Sample Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data) as the Holoscan SDK reference applications.

## Requirements
The only requirement is to make sure the model and data are accessible by the application. At runtime we will need to specify via the `--data` arg, assuming the directory specified contains two subdirectories `endoscopy/` (endoscopy video data directory) and `monai_tool_seg_model/` (model directory).

## Running the application

### Quick start
The easiest way to test this application is to use Holohub CLI from the top level of Holohub

  ```bash
  ./holohub run monai_endoscopic_tool_seg
  ```

### Running the application manually
To run this application, you'll need to configure your PYTHONPATH environment variable to locate the
necessary python libraries based on your Holoscan SDK installation type.

You should refer to the [glossary](../../README.md#Glossary) for the terms defining specific locations within HoloHub.

If your Holoscan SDK installation type is:

* python wheels:

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOHUB_BUILD_DIR>/python/lib
  ```

* otherwise:

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
  ```
Next, run the application, where <DATA_DIR> is a directory that contains two subdirectories `endoscopy/` and `monai_tool_seg_model/`.:

```
python3 tool_segmentation.py --data <DATA_DIR>
```
If you'd like the application to run at the input framerate, change the `replayer` config in the yaml file to `realtime: true`.