# Multi-AI Ultrasound

This application demonstrates how to run multiple inference pipelines in a single application by leveraging the Holoscan Inference module, a framework that facilitates designing and executing inference applications in the Holoscan SDK.

The Multi AI operators (inference and postprocessor) use APIs from the Holoscan Inference module to extract data, initialize and execute the inference workflow, process, and transmit data for visualization.

The applications uses models and echocardiogram data from iCardio.ai. The models include:
- a Plax chamber model, that identifies four critical linear measurements of the heart
- a Viewpoint Classifier model, that determines confidence of each frame to known 28 cardiac anatomical view as defined by the guidelines of the American Society of Echocardiography
- an Aortic Stenosis Classification model, that provides a score which determines likeability for the presence of aortic stenosis

### Requirements

The provided applications are configured to either use the AJA or DELTACAST.TV capture card for input stream, or a pre-recorded endoscopy video (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card or contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on their products.

### Data

[📦️ (NGC) Sample App Data for Multi-AI Ultrasound Pipeline](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_multi_ai_ultrasound_sample_data)

### Build Instructions

Built with the SDK, see instructions from the top level README.

### Run Instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: replayer#' ./apps/multiai/cpp/app_config.yaml \
      && ./apps/multiai/cpp/multiai

    # Python
    python3 ./apps/multiai/python/multiai.py --source=replayer
    ```

* Using an AJA card
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: aja#' ./apps/multiai/cpp/app_config.yaml \
      && ./apps/multiai/cpp/multiai

    # Python
    python3 ./apps/multiai/python/multiai.py --source=aja
    ```

* Using an DELTACAST.TV card
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: videomaster#' ./apps/multiai/cpp/app_config.yaml \
      && ./apps/multiai/cpp/multiai

    # Python
    python3 ./apps/multiai/python/multiai.py --source=videomaster
    ```


> ℹ️ The python app can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment (automatically done by `./run launch`).