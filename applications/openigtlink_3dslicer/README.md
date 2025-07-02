# OpenIGTLink 3D Slicer: Bidirectional Video Streaming with AI Segmentation

This application demonstrates how to interface Holoscan SDK with [3D Slicer](https://www.slicer.org/), using the [OpenIGTLink protocol](http://openigtlink.org/). The application is shown in the application graph below.

![](./images/openigtlink_3dslicer_graph.png)

In summary, the `openigtlink` transmit and receive operators are used in conjunction with an AI segmentation pipeline to:

1. Send Holoscan sample video data from a node running Holoscan SDK, using `OpenIGTLinkTxOp`, to 3D Slicer running on a different node (simulating a video source connected to 3D Slicer):
    * For `cpp` application, the ultrasound sample data is sent.
    * For `python` application, the colonoscopy sample data is sent.
2. Transmit the video data back to Holoscan SDK using OpenIGTLinkIF Module, and receive the data with the `OpenIGTLinkRxOp` operator.
3. Perform an AI segmentation pipeline in Holoscan:
    * For `cpp` application, the ultrasound segmentation model is deployed.
    * For `python` application, the colonoscopy segmentation model is deployed.
4. Use Holoviz in `headless` mode to render image and segmentation and then send the data back to 3D Slicer using the `OpenIGTLinkTxOp` operator.

This workflow allows for sending image data from 3D Slicer over network to Holoscan SDK (running on either `x86` or `arm`), do some compute task (e.g., AI inference), and send the results back to 3D Slicer for visualization. Nodes can run distributed; for example, Holoscan SDK can run on an IGX Orin (Node A) sending the video data, 3D Slicer on a Windows laptop (Node B) and the AI inference pipeline on yet another machine (Node C). Also, note that the `openigtlink` operators can connect to any software/library that supports the OpenIGTLink protocol; here, 3D Slicer is used as it is a popular open source software package for image analysis and scientific visualization.

For the `cpp` application, which does ultrasound segmentations the results look like

![](./images/cpp_ultrasound.png)

and for the `python` application, which does colonoscopy segmentation, the results look like

![](./images/python_colonoscopy.png)

where the image data before Holoscan processing is shown in the left slice view, and the image data with segmentation overlay (after Holoscan processing) is shown in the right slice view.

## Run Instructions

### Machine running 3D Slicer

On the machine running 3D Slicer:

1. In 3D Slicer, open the Extensions Manager and install the `SlicerOpenIGTLink` extension.
2. Next, load the scene `openigtlink_3dslicer/scene/openigtlink_3dslicer.mrb` into 3D Slicer.
3. Go to the `OpenIGTLinkIF` module and make sure that the `SendToHoloscan` connector has the IP address of the machine running Holoscan SDK in the *Hostname* input box (under *Properties*).
4. Then activate the two connectors `ReceiveFromHoloscan` and `SendToHoloscan` (click *Active* check box under *Properties*).

### Machine running Holoscan SDK

On the machine running Holoscan SDK:

1. **Configure the connection**: Update the `host_name` parameters in the configuration files for both `OpenIGTLinkRxOp` operators:
   * `openigtlink_tx_slicer_img`
   * `openigtlink_tx_slicer_holoscan`

   Set these to the IP address of the machine running 3D Slicer.

    > **Note**: This application requires [OpenIGTLink](http://openigtlink.org/) to be installed.

2. **Run the application**: Use the Holohub CLI to launch the application.

* For the `python` application:

    ```sh
    ./holohub run openigtlink_3dslicer --language python
    ```

* For the `cpp` application:

    ```sh
    ./holohub run openigtlink_3dslicer --language cpp
    ```
