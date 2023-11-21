# Hyperspectral Image Segmentation

![](screenshot.png)<br>

This application segments endoscopic hyperspectral cubes into 20 organ classes. It visualizes the result together with the RGB image corresponding to the cube.

The data is a subset of the [HeiPorSPECTRAL](https://www.heiporspectral.org/) dataset. The application loops over the 84 cubes selected. The model is the `2022-02-03_22-58-44_generated_default_model_comparison` checkpoint from [this repository](https://github.com/IMSY-DKFZ/htc), converted to ONNX with the script in `utils/convert_to_onnx.py`.

The data and model are available in [this](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_hyperspectral_segmentation) NGC container.

## Run Instructions

This application requires some python modules to be installed.  For simplicity, a Dockerfile is available.  To generate the container run:
```
./dev_container build --docker_file ./applications/hyperspectral_segmentation/Dockerfile
```
The application can then be built by launching this container and using the provided run script.
```
./dev_container launch
./run build hyperspectral_segmentation
```
Once the application is built it can be launched with the run script.
```
./run launch hyperspectral_segmentation
```

## Viewing Results

With the default settings, the results of this application are saved to `result.png` file in the hyperspectral segmentation app directory. Each time a new image is processed, it overwrites `result.png`.  By opening this image while the application is running, you can see the results as the updates are made (may depend on your image viewer).
