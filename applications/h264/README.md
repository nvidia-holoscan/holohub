# H.264 Encode / Decode Reference Applications

This folder contains two reference applications that showcases the use of H.264
Encode / Decode operators to read, decode H.264 elementary streams and encode,
write H.264 elementary streams to the disk.

## H.264 Endoscopy Tool Tracking Application

The application showcases how to use H.264 video source as input to and output
from the Holoscan pipeline. This application is a modified version of Endoscopy
Tool Tracking reference application in Holoscan SDK that supports H.264
elementary streams as the input and output.

## H.264 Video Decode Application

This is a minimal reference application demonstrating usage of H.264 video
decode operators. This application makes use of H.264 elementary stream reader
operator for reading H.264 elementary stream input and uses Holoviz operator
for rendering decoded data to the native window.

## Building And Running Applications from Holohub Dev Container

### Building Holohub Dev Container

Use the following command from the Holohub root dir to builds the Holohub dev container with
h264 support.

```bash
./dev_container build --docker_file ./applications/h264/Dockerfile
                      --base_img nvcr.io/nvstaging/holoscan/holoscan:v1.0.0.3-dgpu
                      --img holohub:h264
```

### Starting Holohub Dev Container 

Start Holohub dev container from current directory using below command:

```bash
./dev_container launch --img holohub:h264
```

### Building And Running H.264 Applications

Once inside Holohub dev container, follow steps mentioned in README.md from
each application directory to build and run that reference application.
