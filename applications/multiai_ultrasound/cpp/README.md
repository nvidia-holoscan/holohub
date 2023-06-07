# Multi-AI Ultrasound

This application demonstrates how to run multiple inference pipelines in a single application by leveraging the Holoscan Inference module, a framework that facilitates designing and executing inference applications in the Holoscan SDK.

The Inference and the Processing operators use APIs from the Holoscan Inference module to extract data, initialize and execute the inference workflow, process, and transmit data for visualization.

The applications uses models and echocardiogram data from iCardio.ai. The models include:
- a Plax chamber model, that identifies four critical linear measurements of the heart
- a Viewpoint Classifier model, that determines confidence of each frame to known 28 cardiac anatomical view as defined by the guidelines of the American Society of Echocardiography
- an Aortic Stenosis Classification model, that provides a score which determines likeability for the presence of aortic stenosis

The default configuration (`multiai_ultrasound.yaml`) runs on default GPU (GPU-0). Multi-AI Ultrasound application can be executed on multiple GPUs with the Holoscan SDK version 0.6 onwards. A sample configuration file for multi GPU configuration for multi-AI ultrasound application (`mgpu_multiai_ultrasound.yaml`) is present in both `cpp` and `python` applications. The multi-GPU configuration file is designed for a system with at least 2 GPUs connected to the same PCIE network.

### Requirements

The provided applications are configured to either use the AJA capture card for input stream, the YUAN capture card, or a pre-recorded video of the echocardiogram (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for Multi-AI Ultrasound Pipeline](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_multi_ai_ultrasound_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

### Run Instructions

In your `build` directory, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    sed -i -e 's#^source:.*#source: replayer#' applications/multiai_ultrasound/cpp/multiai_ultrasound.yaml
    applications/multiai_ultrasound/cpp/multiai_ultrasound --data <DATA_DIR>/multiai_ultrasound
    ```

* Using a pre-recorded video on multi-GPU system
    ```bash
    sed -i -e 's#^source:.*#source: replayer#' applications/multiai_ultrasound/cpp/mgpu_multiai_ultrasound.yaml
    applications/multiai_ultrasound/cpp/multiai_ultrasound applications/multiai_ultrasound/cpp/mgpu_multiai_ultrasound.yaml --data <DATA_DIR>/multiai_ultrasound
    ```

* Using an AJA card
    ```bash
    sed -i -e 's#^source:.*#source: aja#' applications/multiai_ultrasound/cpp/multiai_ultrasound.yaml
    applications/multiai_ultrasound/cpp/multiai_ultrasound

* Using a YUAN card
    ```bash
    sed -i -e 's#^source:.*#source: qcap#' applications/multiai_ultrasound/cpp/multiai_ultrasound.yaml
    applications/multiai_ultrasound/cpp/multiai_ultrasound
    ```
