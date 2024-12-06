### AprilTagDetection

The `apriltag_detection` extension provides a way to detect [April tags](https://github.com/AprilRobotics/apriltag) of different family. The
detection and processing is done in CUDA.

#### `nvidia::holoscan::ApriltagDetection`

Apriltag Detection codelet

##### Parameters

- **`width`**: Width of the stream (default: None)
  - type: `int`
- **`height`**: Height of the stream (default: None)
  - type: `int`
- **`number_of_tags`**: Number of Apriltags to be detected (default: None)
  - type: `int`
