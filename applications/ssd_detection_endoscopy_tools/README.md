# SSD Detection for Endoscopy Tools

## Model
We can train the [SSD model from NVIDIA DeepLearningExamples repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) with any data of our choosing. Here for the purpose of demonstrating the deployment process, we will use a SSD model checkpoint that is only trained for the [demo video clip](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data).

Please download the models [at this NGC Resource](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/ssd_surgical_tool_detection_model) for `epoch_24.pt`, `epoch24_nms.onnx` and `epoch24.onnx`. You can go through the next steps of Model Conversion to ONNX to convert `epoch_24.pt` into `epoch24_nms.onnx` and `epoch24.onnx`, or use the downloaded ONNX models directly.


### Model Conversion to ONNX
The scripts we need to export the model from .pt checkpoint to the ONNX format are all within this dir `./scripts`. It is a two step process.


 Step 1: Export the trained checkpoint to ONNX. <br> We use [`export_to_onnx_ssd.py`](./scripts/export_to_onnx_ssd.py) if we want to use the model as is without NMS, or [`export_to_onnx_ssd_nms.py`](./scripts/export_to_onnx_ssd_nms.py) to prepare the model with NMS.
 Let's assume the re-trained SSD model checkpoint from the [repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD) is saved as `epoch_24.pt`.
 The export process is
```
# For exporting the original ONNX model
 python export_to_onnx_ssd.py --model epoch_24.pt  --outpath epoch24_temp.onnx
```
```
# For preparing to add the NMS step to ONNX model
python export_to_onnx_ssd_nms.py --model epoch_24.pt  --outpath epoch24_nms_temp.onnx
```
Step 2: modify input shape. <br> Step 1 produces a onnx model with input shape `[1, 3, 300, 300]`, but we will want to modify the input node to have shape `[1, 300, 300, 3]` or in general `[batch_size, height, width, channels]` for compatibility and easy of deployment in the Holoscan SDK. If we want to incorporate the NMS operation in the the ONNX model, we could add a `EfficientNMS_TRT` op, which is documented in [`graph_surgeon_ssd.py`](./scripts/graph_surgeon_ssd.py)'s nms related block.
```
# For exporting the original ONNX model
python graph_surgeon_ssd.py --orig_model epoch24_temp.onnx --new_model epoch24.onnx
```
```
# For adding the NMS step to ONNX model, use --nms
python graph_surgeon_ssd.py --orig_model epoch24_nms_temp.onnx --new_model epoch24_nms.onnx --nms
```

Note that
 - `epoch24.onnx` is used in `ssd_step1.py` and `ssd_step2_route1.py`
 - `epoch24_nms.onnx` is used in `ssd_step2_route2.py` and `ssd_step2_route2_render_labels.py`

## Data
For this application we will use the same [Endoscopy Sample Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data) as the Holoscan SDK reference applications.

## Requirements
There are two requirements
1. To run `ssd_step1.py` and `ssd_step2_route1.py` with the original exported model, we need the installation of PyTorch and CuPy.
<br> To run `ssd_step2_route2.py` and `ssd_step2_route2_render_labels.py` with the exported model with additional NMS layer in ONNX, we need the installation of CuPy.
<br> If you're using the dGPU on the devkit, since there are no prebuilt PyTorch wheels for aarch64 dGPU, the simplest way is to modify the Dockerfile and build from source; if you're on x86 or using the iGPU on the devkit, there should be existing prebuilt PyTorch wheels.
<br> If you choose to build the SDK from source, you can find the [modified Dockerfile here](./docker/Dockerfile) to replace the SDK repo [Dockerfile](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/Dockerfile) to satisfy the installation requirements.
<br> The main changes in Dockerfile for dGPU: the base image changed to `nvcr.io/nvidia/pytorch:22.03-py3` instead of the `nvcr.io/nvidia/tensorrt:22.03-py3` as dGPU's base image; adding the installation of NVTX for optional profiling.
<br>Build the SDK container following the [README instructions](https://github.com/nvidia-holoscan/holoscan-sdk#recommended-using-the-run-script).
<br>
 Make sure the directory containing this application and the directory containing the NGC data and models are mounted in the container. Add the `-v` mount options to the `docker run` command launched by `./run launch` in the SDK repo.

2. Make sure the model and data are accessible by the application.
<br> Make sure the yaml files `ssd_endo_model.yaml` and `ssd_endo_model_with_NMS.yaml` are pointing to the right locations for the ONNX model and data. The assumption in the yaml file is that the `epoch24_nms.onnx` and `epoch24.onnx` are located at:
```
model_file_path: /byom/models/endo_ssd/epoch24_nms.onnx
engine_cache_dir: /byom/models/endo_ssd/epoch24_nms_engines
```
and / or
```
model_file_path: /byom/models/endo_ssd/epoch24.onnx
engine_cache_dir: /byom/models/endo_ssd/epoch24_engines
```
The [Endoscopy Sample Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data) is assumed to be at
```
/workspace/holoscan-sdk/data/endoscopy
```
Please check and modify the paths to model and data in the yaml file if needed.

## Building the application
Please refer to the README under [./app_dev_process](./app_dev_process/README.md) to see the process of building the applications.

## Running the application
Run the incrementally improved Python applications by:
```
python ssd_step1.py

python ssd_step2_route1.py

python ssd_step2_route2.py

python ssd_step2_route2_render_labels.py --labelfile endo_ref_data_labels.csv
```
