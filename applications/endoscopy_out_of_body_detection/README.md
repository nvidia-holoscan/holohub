# Endoscopy Out of Body Detection Application

![Endoscopy Out of Body Detection Workflow](./endoscopy_out_of_body_detection.png)

## Overview

This application performs real-time detection of whether an endoscope is inside or outside the body during endoscopic procedures. For each input frame, the application:

- Classifies the frame as either "in-body" or "out-of-body"
- Provides a confidence score for the classification
- Outputs either to console or to a CSV file (when analytics is enabled)

__Note: This application does not include visualization components.__

## Prerequisites

- NVIDIA Holoscan SDK (version 0.5 or higher)
- CUDA-capable NVIDIA GPU
- CMake build system
- FFmpeg (for data conversion)

## Data Requirements

### Model and Sample Data

The endoscopy detection model and sample datasets are available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/endoscopy_out_of_body_detection). The package includes:

- Pre-trained ONNX model for out-of-body detection: `out_of_body_detection.onnx`
- Sample endoscopy video clips (MP4 format): `sample_clip_out_of_body_detection.mp4`

### Data Preparation

The application requires the input videos to be converted to GXF tensor format. This conversion happens automatically during building, but manual conversion can be done following these steps:

1. Download and extract the data:

```bash
unzip [NGC_DOWNLOAD].zip -d <data_dir>
```

2. Convert video to GXF tensor format using the provided script:

```bash
ffmpeg -i <INPUT_VIDEO_FILE> -fs 900M -pix_fmt rgb24 -f rawvideo pipe:1 | \
python convert_video_to_gxf_entities.py --width 256 --height 256 --channels 3 --framerate 30
```

Note: The conversion script (`convert_video_to_gxf_entities.py`) is available in the [Holoscan SDK repository](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts).

3. Organize the data directory as follows:

```
data/
└── endoscopy_out_of_body_detection/
    ├── LICENSE.md
    ├── out_of_body_detection.onnx
    ├── sample_clip_out_of_body_detection.gxf_entities
    ├── sample_clip_out_of_body_detection.gxf_index
    └── sample_clip_out_of_body_detection.mp4
```

## Configuration

The application uses `endoscopy_out_of_body_detection.yaml` for configuration. Key settings include:

- Input video parameters in the `replayer` section
- Model parameters in the `inference` section
- Analytics settings for data export

### SDK Version Compatibility

For Holoscan SDK version 0.5 or lower, make the following modifications:

In `main.cpp`:

```cpp
// Replace these includes
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/inference_processor/inference_processor.hpp>

// With these
#include <holoscan/operators/multiai_inference/multiai_inference.hpp>
#include <holoscan/operators/multiai_postprocessor/multiai_postprocessor.hpp>

// Replace operator types
ops::InferenceOp -> ops::MultiAIInferenceOp
ops::InferenceProcessorOp -> ops::MultiAIPostprocessorOp
```

In `CMakeLists.txt`:

```cmake
# Update SDK version
set(HOLOSCAN_SDK_VERSION "0.5")

# Replace operator dependencies
holoscan::ops::inference -> holoscan::ops::multiai_inference
holoscan::ops::inference_processor -> holoscan::ops::multiai_postprocessor
```

## Building

Follow the build instructions in the top-level Holohub README.md file.

## Running the Application

### Basic Usage

From your build directory:

```bash
applications/endoscopy_out_of_body_detection/endoscopy_out_of_body_detection \
  --data ../data/endoscopy_out_of_body_detection
```

### Analytics Mode

To enable analytics and export results to CSV:

1. Set `enable_analytics: true` in the configuration file:

```yaml
# endoscopy_out_of_body_detection.yaml
enable_analytics: true
```

2. Configure analytics output (optional):

```bash
# Set output directory (default: current directory)
export HOLOSCAN_ANALYTICS_DATA_DIRECTORY="/path/to/output"

# Set output filename (default: data.csv)
export HOLOSCAN_ANALYTICS_DATA_FILE_NAME="results.csv"
```

The application will create:

- A directory named after the application
- Subdirectories with timestamps for each run
- CSV files containing frame-by-frame classification results

## Output Format

- Console mode: Displays "Likely in-body" or "Likely out-of-body" with confidence scores
- Analytics mode: CSV file with detailed frame information and classification results
