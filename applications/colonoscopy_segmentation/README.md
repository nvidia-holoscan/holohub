# Colonoscopy Polyp Segmentation

Full workflow including a generic visualization of segmentation results from a polyp segmentation models.

### Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded video of the ultrasound data (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI Colonoscopy Segmentation of Polyps](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_colonoscopy_sample_data)

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
    cd colonoscopy_segmentation/python
    python3 colonoscopy_segmentation.py --data=<path_to_datasets>
    ```

* Using an AJA card
    ```bash
    # Python
    export HOLOSCAN_DATA_PATH=<DATA_DIRECTORY>
    cd colonoscopy_segmentation/python
    python3 colonoscopy_segmentation.py --source=aja
    ```
