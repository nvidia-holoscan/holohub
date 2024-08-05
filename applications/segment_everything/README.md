# Segment Everything App


Segment anything allows computing masks or pre-thresholded outputs for objects in natural images. An output mask is created for each query point. In this app the sam encoder and decoder parts are applied to a videostream.

## Model

This application uses a segment anything model (SAM) from https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints

## Setup
The setup is bundled in the [setup.sh](setup.sh) script.

```sh
./applications/segment_everything/setup.sh
```

The setup performs two steps. 
1.  create a pytorch docker container, with a tensorRT version that matches the holoscan tensorRT version. 
2. Then this container creates the tensorRT engine files.

After this we can create the dev_container for the segment_everything app.



## build and launch application container

### Requirements
This application uses a v4l2 compatible device as input.  Please plug in your input device (e.g., webcam) and ensure that `applications/segment_everything/segment_everything.yaml` is set to the corresponding device.  The default input device is set to `/dev/video0`.

Build a holohub container in a new terminal to launch the segment_anything app.
```
cd <HOLOHUB_SOURCE_DIR>
./dev_container build
```
launch the dev container and mount the folder that contains holohub and sam_trt_light
```
./dev_container launch
```

```sh
python applications/segment_everything/segment_one_thing.py
```
or use 
```bash 
./run build segment_everything
./run launch segment_everything
```

