# Multi-AI Ultrasound

This application demonstrates how to run multiple inference pipelines in a single application by leveraging the Holoscan Inference module, a framework that facilitates designing and executing inference applications in the Holoscan SDK.

The Multi AI operators (inference and postprocessor) use APIs from the Holoscan Inference module to extract data, initialize and execute the inference workflow, process, and transmit data for visualization.

The applications uses models and echocardiogram data from iCardio.ai. The models include:

- a Plax chamber model, that identifies four critical linear measurements of the heart
- a Viewpoint Classifier model, that determines confidence of each frame to known 28 cardiac anatomical view as defined by the guidelines of the American Society of Echocardiography
- an Aortic Stenosis Classification model, that provides a score which determines likeability for the presence of aortic stenosis

The default configuration (`multiai_ultrasound.yaml`) runs on the default GPU (GPU 0).

## Multi-GPU requirements and limitations

The sample multi-GPU configuration (`mgpu_multiai_ultrasound.yaml`) assigns inference
models to GPU 0 and GPU 1 from a single application process. It requires at least two
compatible GPUs that CUDA can enumerate together in that process, such as a supported
homogeneous discrete-GPU system with both GPUs on the same PCIe network. Seeing both GPUs
in `nvidia-smi` does not by itself guarantee that the configuration is supported.

The heterogeneous iGPU and dGPU combination on IGX Thor is not supported by this sample.
Through CUDA 13.2, an IGX Thor process cannot enumerate both types of GPU together; CUDA
13.2 supports their concurrent use only from separate processes. On IGX Thor, use the
default single-GPU configuration and select one GPU for the application.

## Requirements

- Python 3.9+
- The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded video of the echocardiogram (replayer). Follow the [AJA Video Systems setup guide](../../../operators/aja_source/setup.md) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for Multi-AI Ultrasound Pipeline](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_multi_ai_ultrasound_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

### Run Instructions

To run this application, you'll need to configure your PYTHONPATH environment variable to locate the
necessary python libraries based on your Holoscan SDK installation type.

You should refer to the [glossary](../../README.md#Glossary) for the terms defining specific locations within HoloHub.

If your Holoscan SDK installation type is:

- python wheels:

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOHUB_BUILD_DIR>/python/lib
  ```

- otherwise:

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
  ```

Next, run the commands of your choice:

- Using a pre-recorded video

    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/multiai_ultrasound/python
    python3 multiai_ultrasound.py --source=replayer --data <DATA_DIR>/multiai_ultrasound
    ```

- Using a pre-recorded video on multi-GPU system

    > **Note:** This command uses both GPUs from one process and is not supported with the
    > IGX Thor iGPU and dGPU combination. See
    > [Multi-GPU requirements and limitations](#multi-gpu-requirements-and-limitations).

    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/multiai_ultrasound/python
    python3 multiai_ultrasound.py --config mgpu_multiai_ultrasound.yaml --source=replayer --data <DATA_DIR>/multiai_ultrasound
    ```

- Using an AJA card

    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/multiai_ultrasound/python
    python3 multiai_ultrasound.py --source=aja
    ```
