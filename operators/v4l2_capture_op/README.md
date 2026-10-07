# V4l2CaptureOp

V4L2 camera capture for Holoscan SDK 5.x on Linux. See the
[module README](../../README.md) for SDK setup and build instructions.

## Reference application

Run these commands from the repository root after completing SDK setup.

```bash
# Build the graph and validate it without touching a device (default mode).
./holoscan_camera run holoscan_camera_v4l2

# Stream from a real device. Requires the device to be mapped into the container.
./holoscan_camera run holoscan_camera_v4l2 capture --language cpp \
    --docker-opts '--device /dev/video0'

# Run CTest through the module CTest driver.
./holoscan_camera test holoscan_camera_v4l2 --language cpp
```

The mode is a positional argument, and application flags must travel through `--run-args`
because the CLI rejects unrecognized arguments of its own. The `capture` mode already supplies
`--capture --frames 60`, so `--run-args` is only needed to override defaults:

```bash
./holoscan_camera run holoscan_camera_v4l2 capture --language cpp \
    --docker-opts '--device /dev/video2' \
    --run-args '--device /dev/video2 --frames 300 --timeout 30'
```

`--memory-kind {host|pinned|device}` selects the frame placement described under
[Frame placement](#frame-placement), applying it to the source *and* the reference sink so the two
port contracts agree. Because the placement is settled when the plan compiles, the default
`--validate` mode exercises it with no camera attached:

```bash
# Compiles the plan with a device placement and reports it; opens no device.
./holoscan_camera run holoscan_camera_v4l2 --language cpp \
    --run-args '--memory-kind device'
```

## Interface

| Operator | Implementation | Ports | Parameters |
| --- | --- | --- | --- |
| `V4l2CaptureOp` | C++, memory-mapped V4L2 capture | `frame`: `holoscan::schema::ImageT`, rank-2 `uint8` tensor of shape `[height, 2 * width]`, placed per `memory_kind` | `device`, `width`, `height`, `fps`, `frame_id`, `memory_kind` |

`V4l2CaptureOp` opens `/dev/video*`, negotiates YUYV, streams via `mmap`, and publishes
frames from a reader thread that wakes the operator through a notification source.

The published frame is the **packed YUYV 4:2:2 frame**, chroma included, under
`ImageEncoding_YUYV`. The tensor is rank 2 and one byte per element, shaped `[height, 2 * width]`:
a row is its own byte extent, and the pixel geometry is read from the descriptor's `width`,
`height`, and `encoding` rather than inferred from the shape. A downstream consumer must declare
that carriage, including the placement the operator was constructed with:

```cpp
spec.input(frame_in, "frame_in")
    .queue_depth(4U)
    .expects_tensor(holoscan::TensorInputSpec{
        .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                           .dtype = holoscan::holoscan_camera::kFrameElementDtype,
                           .rank = 2U},
    });
```

Capture width must be even. A packed 4:2:2 group spans two columns, and an even width is what
makes the driver's `2 * width` row equal the `4 * ceil(width / 2)` the schema derives, so the
tensor addresses exactly the frame the descriptor describes.

A consumer whose declared rank, dtype, or placement differs from what the operator publishes will
fail the port contract at graph authoring time. The rank and dtype are fixed by the payload above;
the placement is the operator's `memory_kind`, so the two must be chosen together.

## Frame placement

The `memory_kind` constructor parameter selects where the frame lands. It is frozen into the port
contract at `setup()`, so it is an authoring-time choice rather than something a running graph
renegotiates, and a consumer's `expects_tensor` must name the same placement.

| `memory_kind` | Frame lands in | Use when | Needs a device binding |
| --- | --- | --- | --- |
| `kHost` (default) | Pageable host memory | Consumers read on the CPU, and nothing downstream needs the GPU | No — must be absent |
| `kPinnedHost` | Page-locked host memory | Consumers read on the CPU *and* something downstream transfers to the GPU, which then becomes a direct DMA instead of a staged copy | No — must be absent |
| `kCudaDevice` | CUDA device memory | Every consumer is device-resident, so paying the transfer here avoids a round trip | **Yes** |

The driver always delivers into a pageable `mmap` buffer, so every placement copies once out of it;
the parameter chooses what that copy targets, not whether it happens. Use it to state where this
operator's own consumers want the data. Repairing a placement mismatch *between* two operators is
an edge concern — see `ConnectionOptions::memory_conversion` — not a reason to change the producer.

`kUnknown` and `kCudaManaged` are rejected at construction.

### `kCudaDevice` requires the application to name a GPU

Compilation will not choose a device for a device-resident pool; it rejects the plan with
`DEVICE_UNRESOLVED`, because an implicitly chosen GPU is an implicitly chosen cross-device transfer
later. The application supplies it, since which GPU to use is a deployment fact the operator does
not own:

```cpp
holoscan::CompileOptions options{};
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "camera", .output_port = "frame", .device = holoscan::DeviceId{0}});
const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
```

The binding must be *absent* for `kHost` and process-local `kPinnedHost`, where supplying one is
rejected as `TENSOR_DEVICE_BINDING_UNEXPECTED`. It is therefore not safe to set unconditionally —
see `--memory-kind` handling in the reference application for the conditional form.

## C++ Usage

```cpp
#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <v4l2_capture_op/v4l2_capture_op.hpp>

namespace mm = holoscan::holoscan_camera;

holoscan::Graph graph{"example"};
auto camera = graph.op<mm::V4l2CaptureOp>("camera", "/dev/video0", 640, 480, 30);
```

## Capture counters

A capture run ends with one summary line:

```text
frames: received=N with-capture-time=N degraded=N corrupt=N unnamed-clock=N
```

Use these counters to check completion, timestamp coverage, and sample quality:

| Counter | Meaning |
| --- | --- |
| `received` | Number of frames received by the reference sink. Compare with the requested `--frames`. |
| `with-capture-time` | Number of received frames carrying a projected capture timestamp. |
| `degraded` | `SampleFlags::kDegraded`. Set when the device sequence skipped forward (the driver produced frames that were never collected), when the device sequence went backwards, or when the frame arrived shorter than its declared byte extent. The rows that did arrive are still real measurements. |
| `corrupt` | `SampleFlags::kInvalid`. Set only when the driver returned the buffer with `V4L2_BUF_FLAG_ERROR` — the driver reporting its own contents as bad. The pixels are not measurements and must not be treated as such. |
| `unnamed-clock` | The driver did not name the clock its timestamp came from, so the operator declined to project a capture time rather than invent a correspondence. |

Reject samples carrying `SampleFlags::kInvalid`; inspect `SampleFlags::kDegraded`
when deciding whether a partial frame or sequence discontinuity is acceptable.
Check `received` against the requested frame count and `with-capture-time`
against `received` to assess completion and timestamp coverage.

See the [V4L2 benchmarks and qualification results](../../docs/v4l2_capture_op_benchmarks.md)
for recorded platform behavior and the [release notes](../../RELEASE_NOTES.md)
for known issues.
