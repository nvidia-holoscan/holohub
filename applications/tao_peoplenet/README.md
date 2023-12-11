# TAO PeopleNet Detection Model on V4L2 Video Stream

Use the TAO PeopleNet available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) to detect faces and people in a V4L2 supported video stream. HoloViz is used to draw bounding boxes around the detections.

**Prerequisite**: Download the PeopleNet ONNX model from the NGC website:
```sh
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx' -O applications/tao_peoplenet/resnet34_peoplenet_int8.onnx
```

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

## Run Instructions

Run with:
```sh
./run launch tao_peoplenet
```