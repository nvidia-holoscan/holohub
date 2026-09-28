# V4L2 inference and Holoviz

A C++ Holoscan **5.x Early Access** application using the camera module's
`V4l2CaptureOp`, the inference module's `TensorPreprocessorOp` and `InferenceOp`
(HoloInfer/TensorRT), and the visualization module's `HolovizOp`.

```text
V4l2CaptureOp.frame (ImageT, CUDA YUYV)
    -> TensorPreprocessorOp.camera_frame
       -> model_input (CUDA float32 NCHW)
          -> InferenceOp.model_input
             -> prediction (CUDA float32 NCHW)
                -> TensorToImageOp (RGB8 ImageT adapter)
                   -> HolovizOp.prediction (window)
```

The additional adapter is necessary because HoloInfer emits a `Tensor` and the
5.x Holoviz image port consumes `schema::ImageT`. It checks shape and strides,
interleaves the three RGB planes, converts normalized floats to uint8, and
supplies the RGB8 image descriptor.
Inference executes on the GPU and emits CUDA tensors. The adapter downloads its
input through `Tensor::copy_to_host()` into a reusable CPU buffer, respecting
producer readiness. Holoviz uploads the resulting image for display. This
sample favors a simple adapter over a fully GPU-resident display path.

## Dependencies

- Linux, C++20 compiler, CMake, CUDA 13, TensorRT, and NVIDIA Vulkan drivers.
- A V4L2 camera supporting YUYV at the requested size and frame rate.
- A working display for capture mode.
- Holoscan 5.x installed with `sensor_io`, `inference`, and `visualization`.
- The matching `holoscan-camera` package, version 0.3.0 or newer, exporting
  `holoscan::v4l2_capture_op`.
- The inference installation must also export `holoscan::op_tensor_preprocessor`
  and its `input_kind="image"` / `image_input_port()` API.

These APIs are under development and are **not supplied by HoloHub's default
Holoscan 4.6 image**. Source interfaces were inspected at:

| Component | Source revision |
| --- | --- |
| SDK core, TensorPreprocessor, HoloInfer, visualization | `main-5x`, `1ecc161b6726340d9543eed8bc30e49543ed218f` |
| Camera module (pinned by that SDK) | `931b9c0ca7f76b8228aa7f7f98946f588466e109` |

TensorPreprocessor is part of `main-5x`, merged on September 23, 2026. Build the
camera package against the same SDK installation. Configuration fails clearly
if the preprocessor target is missing. The app does not download or modify
SDK/module sources.

## Model

This app uses the inference module's existing example **`identity_model.onnx`**
unchanged. It has one float32 input named `input` and one float32 output named
`output`, both shaped `[1, 3, 256, 256]` (NCHW). Its single ONNX `Identity` node
returns the input unchanged. The preprocessor resizes camera frames to 256x256
and normalizes RGB to `[0,1]`; the displayed image retains those colors.

The model comes from `public/modules/inference/examples/models/identity_model.onnx`
in the SDK revision listed above (or `modules/inference/...` in the SDK public
source root). Its SHA-256 is
`0d26171c20a21c09f9103d3b355cb32d1ae181273b5d3f546d5bfacd78c5ec15`.
The SDK examples package also installs it under
`<prefix>/<libdir>/holoscan/examples/inference/models/identity_model.onnx`.
It is a pipeline demonstration with no accuracy claim.

From the HoloHub root, copy the SDK example into the ignored data directory.
Set the source path to your SDK checkout or examples installation:

```bash
export V4L2_IDENTITY_MODEL=/path/to/sdk/public/modules/inference/examples/models/identity_model.onnx
mkdir -p data/v4l2_inference_holoviz
cp -n "$V4L2_IDENTITY_MODEL" data/v4l2_inference_holoviz/identity_model.onnx
sha256sum data/v4l2_inference_holoviz/identity_model.onnx
```

The copy preserves any existing destination; verify the checksum above. The
model is not generated or included in this application's source. Models and
generated TensorRT engines belong in ignored `data/` or another writable data
directory; HoloInfer caches engines beside the ONNX file. First-run engine
building is included in the runtime deadline; increase `--timeout` if necessary.

To use another model, pass `--model`, `--model-width`, and `--model-height`.
Keep the default 256x256 dimensions for the SDK identity model. This app supports
**one image-to-image input/output pair**, contiguous FP32 NCHW `[1, 3, H, W]`,
matching input/output dimensions, RGB order and `[0,1]`
normalization. HoloInfer maps the model's single input and output to the graph
port names `model_input` and `prediction`; ONNX binding names may differ.
Classifiers, detectors, segmentation logits, NHWC models, or models requiring
other normalization need corresponding preprocessing and output adaptation.

The preprocessor uses BT.601/full-range YUYV conversion. Its inspected revision
does not support BT.709 or limited-range YUYV conversion. The tested Logitech
BRIO reports limited-range YUYV, so the smoke test verifies pipeline operation
and visible camera imagery, not correct black/white levels or color accuracy.
Use a matching camera mode or add limited-range conversion before relying on
the displayed colors. Resizing stretches the frame to the model dimensions.

## Build and run

HoloHub normally obtains Holoscan from the SDK development container used as
`BASE_IMAGE`; applications locate the installed libraries with CMake
`find_package(holoscan)`. No separate host SDK installation is needed for a
container build. This checkout defaults to SDK 4.6, so this 5.x application needs
an explicitly selected compatible SDK image. An `holoscan-sdk-build-*` image
contains tools for building the SDK and may not contain an installed SDK.
The `minimum_required_version` metadata field does not install or upgrade it.

Prepare a development image containing the compatible SDK and camera package,
their CMake package paths and compiler. The app's Dockerfile adds the CLI
version pinned by the repository wrapper and Python `onnx` for model verification.
Substitute the image's actual tag below:

```bash
export V4L2_SDK_IMAGE=your-holoscan-5x-camera-dev:latest
export DISPLAY=:1
export XDG_SESSION_TYPE=x11

./holohub build v4l2_inference_holoviz --language cpp \
    --base-img "$V4L2_SDK_IMAGE" --dryrun --verbose
./holohub build v4l2_inference_holoviz --language cpp \
    --base-img "$V4L2_SDK_IMAGE" --verbose

./holohub run v4l2_inference_holoviz validate --language cpp \
    --base-img "$V4L2_SDK_IMAGE" --dryrun --verbose
./holohub run v4l2_inference_holoviz validate --language cpp \
    --base-img "$V4L2_SDK_IMAGE" --verbose

./holohub run v4l2_inference_holoviz capture --language cpp \
    --base-img "$V4L2_SDK_IMAGE" --dryrun --verbose
./holohub run v4l2_inference_holoviz capture --language cpp \
    --base-img "$V4L2_SDK_IMAGE" --verbose
```

`capture` maps `/dev/video0` into the container. For another device, override
both the mapping and the application option (preview the same command first):

```bash
./holohub run v4l2_inference_holoviz capture --language cpp \
    --base-img "$V4L2_SDK_IMAGE" --dryrun --verbose \
    --docker-opts="--device=/dev/video2" \
    --run-args="--device /dev/video2 --width 1280 --height 720 --frames 300 --timeout 180"
```

Remove `--dryrun` to execute after inspecting the preview. For an existing
local SDK development environment, the corresponding `--local` commands work
with `CMAKE_PREFIX_PATH` including both the SDK and camera installation prefixes.

`--validate` compiles the complete graph without starting its operators. It
does not test camera access, engine building, or rendered pixels. CUDA device
placement is part of graph compilation, so a usable CUDA installation may
still be required. All stages use visible CUDA device 0; select a physical GPU
with `CUDA_VISIBLE_DEVICES` in the execution environment.

On the development host, display `:1` has working X11 authorization. The HoloHub
wrapper forwards `DISPLAY`, the X11 socket, and the existing Xauthority cookie
to the container. Select your actual display on other hosts. Check access with
`xdpyinfo -display "$DISPLAY"`; if it succeeds, no cookie replacement or X server
package installation is needed.

### Validated local build

On September 28, 2026, this application was built against the exact SDK and
camera revisions listed above, verified against their remote `main-5x` and
`main` tips. The SDK source was used without patches or a preprocessor overlay;
Camera 0.5.0 was built against that installation. The existing SDK and camera
working copies were preserved, including local camera edits.

The prepared development image is
`holohub:v4l2-inference-holoviz-main5x-1ecc161b6`, with the SDK and camera installed
under `/opt/nvidia/holoscan`. The host staging prefix is
`install/v4l2-sdk5-main5x`. The executable is under
`build/v4l2_inference_holoviz/applications/v4l2_inference_holoviz/`.

The four CTest checks (help, graph compilation, CPU conversion/options, and the
SDK identity model) passed. A Logitech BRIO on `/dev/video0`, capturing YUYV
640x480 at a requested 30 FPS, produced 120 inferred images on an RTX A6000:

```text
inference frames: requested=120 observed=120 result=PASS
```

The Holoviz window on X11 display `:1` was captured and inspected to confirm
camera imagery. The count measures adapter emissions, not display presentations
or achieved frame rate. The color-range limitation above still applies.

To rebuild and repeat this run using the prepared local image:

```bash
export DISPLAY=:1
export XDG_SESSION_TYPE=x11
export V4L2_APP_IMAGE=holohub:v4l2-inference-holoviz-main5x-1ecc161b6
./holohub run v4l2_inference_holoviz capture --language cpp \
    --img "$V4L2_APP_IMAGE" --no-docker-build --dryrun --verbose
./holohub run v4l2_inference_holoviz capture --language cpp \
    --img "$V4L2_APP_IMAGE" --no-docker-build --verbose
```

## Runtime behavior and checks

The camera source uses its device notifications. Preprocessing, inference, and
the adapter run on arriving samples. Holoviz renders on image arrival in its
own partition. Connections use depth-one drop-oldest queues, allowing slow
inference or display to discard old frames without blocking capture. CUDA
output pools for camera, preprocessor, and inference are explicitly bound to visible device 0.
Host output pools have no CUDA device binding. The graph uses a realtime clock.

After `--frames` successful adapter emissions (120 by default), or `--timeout`
seconds (120 by default), the app requests cooperative shutdown and prints:

```text
inference frames: requested=120 observed=120 result=PASS
```

Runtime failure or an insufficient count returns nonzero. This count measures
inferred images emitted toward Holoviz, not displayed frames: the display
mailbox can drop frames and shutdown may precede the last presentation. Verify
the window visually for the resized camera image with preserved colors. Closing
it before the requested count completes the run with a failed count. The timeout requests cooperative
stop; it cannot forcibly interrupt a blocked driver or TensorRT engine build.

The CPU-only checks cover argument validation, rejection of incompatible model
layouts/strides, planar-to-interleaved RGB order, clipping, rounding, and
non-finite output. They can run without CUDA, the SDK, a camera, or a display:

```bash
cmake -S applications/v4l2_inference_holoviz -B /tmp/v4l2-inference-check \
    -DV4L2_INFERENCE_HOLOVIZ_BUILD_APP=OFF -DBUILD_TESTING=ON
cmake --build /tmp/v4l2-inference-check
ctest --test-dir /tmp/v4l2-inference-check --output-on-failure
```

When Python `onnx` and the model are available, CTest also registers the identity
model test. It checks the SDK model's bindings, shape and type, and verifies
numerically that its output equals its input. For a model outside the default
data directory, configure `-DV4L2_INFERENCE_HOLOVIZ_TEST_MODEL=/path/to/identity_model.onnx`.
To run the test directly in the prepared container or development environment:

```bash
python3 applications/v4l2_inference_holoviz/tests/test_identity_model.py \
    --model data/v4l2_inference_holoviz/identity_model.onnx
```

The CPU/model checks alone do not establish end-to-end operation. For another
compatible SDK environment, also use `./holohub test
v4l2_inference_holoviz --language cpp` (preview with `--dryrun --verbose`) and
the validation/capture commands above.
