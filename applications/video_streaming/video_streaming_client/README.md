# Video Streaming Client Demo

This is the enhanced video streaming client demo application that supports both V4L2 camera input and video file replay.

## Features

- **V4L2 Camera Support**: Capture live video from webcam
- **Video File Replay**: Play back pre-recorded video files
- **Bidirectional Streaming**: Send and receive video frames
- **HoloViz Visualization**: Real-time video display
- **Configurable Resolution**: Support for different video resolutions

## Running the Application

> [!IMPORTANT] The client applications requires Holoscan SDK 3.5.0. Either set the SDK version environment variable before running the applications, or use the `--base-img` option to specify the base image.
>
> ```bash
> # Set SDK version environment variable
> export HOLOHUB_BASE_SDK_VERSION=3.5.0
> ```

> [!NOTE] The client requires OpenSSL 3.4.0, which is installed inside the custom Dockerfile.

### V4L2 Camera (Webcam)

```bash
./holohub run video_streaming client_v4l2
```

### Video File Replay

```bash
./holohub run video_streaming client_replayer
```

## Configuration Files

- `streaming_client_demo.yaml`: Default configuration for V4L2 camera (640x480)
- `streaming_client_demo_replayer.yaml`: Configuration for video file replay (854x480)

## Dependencies

- Holoscan SDK 3.5.0+
- video_streaming operator
- OpenCV
- CUDA 12.x (currently not working with CUDA 13.x)
- OpenSSL 3.4.0 (installed inside the custom Dockerfile)
