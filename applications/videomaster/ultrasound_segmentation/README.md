# Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model. 

### Requirements

The provided applications are configured to either use the AJA or DELTACAST.TV capture card for input stream, or a pre-recorded endoscopy video (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card or contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on their products.

### Data

[📦️ (NGC) Sample App Data for AI-based Bone Scoliosis Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_ultrasound_sample_data)

### Build Instructions

Built with the SDK, see instructions from the top level README.

### Run Instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: replayer#' ./apps/ultrasound_segmentation/cpp/app_config.yaml \
      && ./apps/ultrasound_segmentation/cpp/ultrasound_segmentation

    # Python
    python3 ./apps/ultrasound_segmentation/python/ultrasound_segmentation.py --source=replayer
    ```

* Using an AJA card
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: aja#' ./apps/ultrasound_segmentation/cpp/app_config.yaml \
      && ./apps/ultrasound_segmentation/cpp/ultrasound_segmentation

    # Python
    python3 ./apps/ultrasound_segmentation/python/ultrasound_segmentation.py --source=aja
    ```

* Using an DELTACAST.TV card
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: videomaster#' ./apps/ultrasound_segmentation/cpp/app_config.yaml \
      && ./apps/ultrasound_segmentation/cpp/ultrasound_segmentation

    # Python
    python3 ./apps/ultrasound_segmentation/python/ultrasound_segmentation.py --source=videomaster
    ```

> ℹ️ The python app can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment (automatically done by `./run launch`).