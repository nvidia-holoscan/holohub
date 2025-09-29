# Custom GStreamer Sink Tutorial

This tutorial demonstrates how to create a native GStreamer sink element that depends only on GStreamer libraries.

## Overview

The custom sink (`simplecustomsink`) is a video sink that:
- Processes incoming video frames
- Prints frame information to console
- Optionally saves frames to disk as raw files
- Demonstrates proper GStreamer element implementation

## Files

**In `operators/gstreamer/`:**
- `gstsimplecustomsink.h` - Header file defining the sink element
- `gstsimplecustomsink.c` - Implementation of the custom sink
- `gst_sink_resource.hpp` - Holoscan Resource wrapper for the GStreamer sink

**In `tutorials/gstreamer/`:**
- `custom-sink-example.cpp` - Example program using the custom sink
- `holoscan_gst_example.cpp` - Holoscan application using GstSinkResource
- `README_custom_sink.md` - This documentation

## Building

### Prerequisites
- GStreamer 1.0 development libraries
- CMake 3.16 or later
- C/C++ compiler

### Ubuntu/Debian
```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### Build Commands
```bash
cd /workspace/holohub/tutorials/gstreamer
mkdir -p build
cd build
cmake ..
make
```

**Note:** The build process automatically builds the GStreamer sink library from `operators/gstreamer/core/` and links it with the tutorial applications.

## Usage

### Basic Usage
```bash
./custom-sink-example
```
This will:
- Create a pipeline with `videotestsrc -> videoconvert -> simplecustomsink`
- Display SMPTE color bars
- Print frame information to console
- Run for 300 frames (10 seconds at 30 fps)

### Save Frames to Files
```bash
./custom-sink-example --save-frames
```
Saves frames to `/tmp/frame_XXXXXX.raw`

### Custom Output Directory
```bash
./custom-sink-example --save-frames /path/to/output/dir
```

### Holoscan Integration Example

**Basic usage:**
```bash
./holoscan-gst-example
```

**With custom parameters:**
```bash
# Run for 150 iterations with snow pattern
./holoscan-gst-example --count 150 --pipeline "videotestsrc pattern=1 ! videoconvert"

# Test with audio (shows universal sink capability)
./holoscan-gst-example -c 50 -p "audiotestsrc ! audioconvert"

# Use camera input (if available)
./holoscan-gst-example --pipeline "autovideosrc ! videoconvert"
```

**Command line options:**
- `-c, --count <number>`: Number of iterations to run (default: 300)
- `-p, --pipeline <desc>`: GStreamer pipeline description (default: videotestsrc pattern=0 ! videoconvert)  
- `-h, --help`: Show help message

This demonstrates how to use the `GstSinkResource` within a Holoscan application with configurable parameters.

## Custom Sink Properties

The `simplecustomsink` element supports the following properties:

- `save-frames` (boolean): Enable/disable frame saving to files
- `output-dir` (string): Directory to save frames (default: "/tmp")
- `fps` (double): Target FPS for display purposes (default: 30.0)

### Using in GStreamer Pipelines

You can use the custom sink in any GStreamer pipeline:

```bash
# Using gst-launch-1.0 (after installing the plugin)
gst-launch-1.0 videotestsrc num-buffers=100 ! videoconvert ! simplecustomsink save-frames=true output-dir=/tmp

# In your own application
pipeline = gst_parse_launch("videotestsrc ! videoconvert ! simplecustomsink name=mysink", NULL);
sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
g_object_set(sink, "save-frames", TRUE, "output-dir", "/path/to/frames", NULL);
```

## Implementation Details

### Element Structure
The sink inherits from `GstBaseSink` which provides:
- Buffer queuing and synchronization
- State management
- Quality of Service (QoS) handling

### Key Methods Implemented
- `set_caps()`: Called when input format is negotiated
- `render()`: Called for each incoming buffer/frame
- `start()/stop()`: Called during state transitions

### Supported Video Formats
- RGB variants: RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR
- YUV formats: I420, YV12, NV12, NV21, YUY2, UYVY

## Extending the Sink

To extend functionality, you can:

1. **Add new properties**: Modify the property enum and add corresponding getters/setters
2. **Process frames differently**: Modify the `render()` method
3. **Add new video formats**: Update the pad template capabilities
4. **Add threading**: Use GStreamer's base classes or implement custom threading

### Example Extensions
```c
// Add a new property for frame processing
enum {
    PROP_0,
    PROP_SAVE_FRAMES,
    PROP_OUTPUT_DIR,
    PROP_FPS,
    PROP_PROCESS_FRAMES,  // New property
};

// In render() method, add frame processing
if (custom_sink->process_frames) {
    // Apply filters, transformations, etc.
    process_video_frame(map.data, custom_sink->width, custom_sink->height);
}
```

## Debugging

Enable GStreamer debug output:
```bash
GST_DEBUG=simplecustomsink:5 ./custom-sink-example
```

Debug levels:
- 1: ERROR
- 2: WARNING  
- 3: INFO
- 4: DEBUG
- 5: TRACE

## Plugin Registration

The current implementation registers the element directly in the application. To create a proper GStreamer plugin that can be loaded dynamically:

1. Implement `gst_plugin_init()` properly
2. Use `GST_PLUGIN_DEFINE()` macro
3. Compile as a shared library
4. Install in GStreamer plugin directory

## Troubleshooting

### Build Issues
- Ensure all GStreamer development packages are installed
- Check PKG_CONFIG_PATH includes GStreamer .pc files
- Verify CMake finds all required libraries

### Runtime Issues  
- Check element registration succeeded
- Verify input format is supported
- Enable debug output for detailed information
- Ensure output directory exists and is writable

## Holoscan Resource Integration

The `GstSinkResource` class provides a Holoscan Resource wrapper around the custom GStreamer sink, making it easy to use within Holoscan applications.

### Key Features of GstSinkResource

- **Holoscan Resource API**: Follows Holoscan Resource patterns with `initialize()` and proper lifecycle management
- **Automatic GStreamer Management**: Handles GStreamer initialization and element registration
- **Element Access**: Provides `get_element()` method to access the sink element for pipeline integration
- **Property Management**: Methods to configure sink properties (save buffers, output directory, data rate)
- **Resource Cleanup**: Automatic cleanup of GStreamer resources in destructor

### Usage in Holoscan Applications

The `GstSinkResource` follows the same parameter pattern as other Holoscan resources (like ROS2 Bridge):

```cpp
#include "../../operators/gstreamer/gst_sink_resource.hpp"

// In your operator class:
class MyGstOperator : public holoscan::Operator {
 public:
  void setup(OperatorSpec& spec) override {
    // Add GstSinkResource as a parameter (similar to ros2_bridge pattern)
    spec.param(gst_sink_resource_, "gst_sink_resource", "GStreamerSink", 
               "GStreamer sink resource object");
  }

  void initialize() override {
    Operator::initialize();
    // Ensure the resource is provided and valid
    assert(gst_sink_resource_.get());
    assert(gst_sink_resource_.get()->valid());
    
    // Create your own pipeline and add the sink element
    pipeline_ = gst_pipeline_new("my-pipeline");
    GstElement* source = gst_element_factory_make("videotestsrc", "source");
    GstElement* convert = gst_element_factory_make("videoconvert", "convert");
    GstElement* sink = gst_sink_resource()->get_element();
    
    gst_bin_add_many(GST_BIN(pipeline_), source, convert, sink, nullptr);
    gst_element_link_many(source, convert, sink, nullptr);
  }

 protected:
  GstSinkResourcePtr gst_sink_resource() { return gst_sink_resource_.get(); }

 private:
  Parameter<GstSinkResourcePtr> gst_sink_resource_;
  GstElement* pipeline_ = nullptr;
};

// In your application:
class MyApp : public Application {
 public:
  void compose() override {
    // Create the resource outside the operator
    auto gst_sink = make_resource<GstSinkResource>("gst_sink", "sink_name", false, "/tmp", 30.0);
    
    // Pass it to the operator as a parameter
    auto my_op = make_operator<MyGstOperator>(
        "my_op",
        Arg("gst_sink_resource", gst_sink)
    );
    
    add_operator(my_op);
  }
};
```

This provides seamless integration between GStreamer and Holoscan, allowing you to leverage GStreamer's powerful media processing capabilities within Holoscan's application framework. The `GstSinkResource` focuses solely on managing the sink element, giving you full control over pipeline construction and element connectivity.

## Next Steps

1. Try modifying the frame processing in `render()`
2. Add new properties for different behaviors
3. Implement proper plugin loading
4. Add support for audio sinks using `GstAudioBaseSink`
5. Explore using `GstSinkResource` in more complex Holoscan applications
