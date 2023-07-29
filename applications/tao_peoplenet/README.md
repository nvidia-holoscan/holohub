# TAO PeopleNet Detection Model on V4L2 Video Stream

Uses the [TAO PeopleNet available on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) to detect faces and people in a V4L2 supported video stream. HoloViz is used to draw bounding boxes around the detections.

## Requirements

### Containerized Development

If using a container outside the Holoscan SDK `run` script, add `--group-add video` and `--device /dev/video0:/dev/video0` (or the ID of whatever device you'd like to use) to the `docker run` command to make your camera device available in the container.

### Local Development

Install the following dependency:
```sh
sudo apt-get install libv4l-dev=1.18.0-2build1
```

If you do not have permissions to open the video device, do:
```sh
 sudo usermod -aG video $USER
```

### PeopleNet Model

[Download the model from NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) and put in a folder `DATA_DIRECTORY`.

### TensorRT Conversion

[Download the TAO Converter app](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter) **for your platform** and convert the PeopleNet model to a TensorRT engine. 

If you download the `tao-converter` to the folder `DATA_DIRECTORY` the conversion can be run by:
```sh
cd <DATA_DIRECTORY>
chmod +x tao-converter
./tao-converter \
    -k tlt_encode \
    -d 3,544,960 \
    -w 40960M \
    -t fp16 \
    -o output_cov/Sigmoid,output_bbox/BiasAdd \
    -e engine.trt \
    model.etlt
```
This command will generate the TensorRT engine `engine.trt`, which is given as input to the app.

## Run Instructions

First, set the environment variable to the absolute path of the folder containing the location of the TensorRT engine:
```sh
export HOLOSCAN_DATA_PATH=<DATA_DIRECTORY>
```

Next, build with:
```sh
./run build tao_peoplenet
```
then run with:
```sh
./run launch tao_peoplenet
```