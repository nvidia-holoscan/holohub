# Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model. 

### Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded video of the ultrasound data (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Bone Scoliosis Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_ultrasound_sample_data)

```
unzip holoscan_ultrasound_sample_data_20220608.zip -d <data_dir>
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
    cd ultrasound_segmentation/python
    python3 ultrasound_segmentation.py --source=aja
    ```

* Using an AJA card
    ```bash
    # Python
    export HOLOSCAN_DATA_PATH=<DATA_DIRECTORY>
    cd ultrasound_segmentation/python
    python3 ultrasound_segmentation.py --source=aja
    ```
