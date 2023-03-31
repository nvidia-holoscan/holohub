# Multi-AI Ultrasound

This application demonstrates how to run multiple inference pipelines in a single application by leveraging the Holoscan Inference module, a framework that facilitates designing and executing inference applications in the Holoscan SDK.

The Multi AI operators (inference and postprocessor) use APIs from the Holoscan Inference module to extract data, initialize and execute the inference workflow, process, and transmit data for visualization.

The applications uses models and echocardiogram data from iCardio.ai. The models include:
- a Plax chamber model, that identifies four critical linear measurements of the heart
- a Viewpoint Classifier model, that determines confidence of each frame to known 28 cardiac anatomical view as defined by the guidelines of the American Society of Echocardiography
- an Aortic Stenosis Classification model, that provides a score which determines likeability for the presence of aortic stenosis

### Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded video of the echocardiogram (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for Multi-AI Ultrasound Pipeline](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_multi_ai_ultrasound_sample_data)

```
unzip holoscan_multi_ai_ultrasound_sample_data_20221201.zip -d <data_dir>
```

### Run Instructions

* (Optional) Create and use a virtual environment:

  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* Install Holoscan PyPI package

  ```
  pip install holoscan
  ```

Run the commands of your choice:

* Using a pre-recorded video
    ```bash
    # Python
    export HOLOSCAN_DATA_PATH=<DATA_DIRECTORY>
    export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
    python3 multiai_ultrasound.py --source=replayer
    ```

* Using an AJA card
    ```bash
    # Python
    export HOLOSCAN_DATA_PATH=<DATA_DIRECTORY>
    export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
    python3 multiai_ultrasound.py --source=aja
    ```

> ℹ️ The python app can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment.
