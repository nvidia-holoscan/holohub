# Video Streaming Demo Client Enhanced

This is the enhanced video streaming client demo application that supports both V4L2 camera input and video file replay.

## Features

- **V4L2 Camera Support**: Capture live video from webcam
- **Video File Replay**: Play back pre-recorded video files
- **Bidirectional Streaming**: Send and receive video frames
- **HoloViz Visualization**: Real-time video display
- **Configurable Resolution**: Support for different video resolutions

## Running the Application

**⚠️ Important: Enhanced applications require Holoscan SDK 3.5.0. Use the base image parameter to ensure compatibility.**

### V4L2 Camera (Webcam)
```bash
# From holohub root directory - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_client --language cpp
```

### Video File Replay
```bash
# From holohub root directory - with Holoscan 3.5.0 base image
./holohub run --docker-opts='-e EnableHybridMode=1' --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.5.0-dgpu video_streaming_demo_client --language cpp --run-args='-c streaming_client_demo_replayer.yaml'
```

## Configuration Files

- `streaming_client_demo.yaml`: Default configuration for V4L2 camera (640x480)
- `streaming_client_demo_replayer.yaml`: Configuration for video file replay (854x480)

## Dependencies

- Holoscan SDK 3.5.0+
- video_streaming operator
- OpenCV
- CUDA Toolkit
