# TapNext Inference Extension

This GXF extension runs TensorRT inference for TapNext-style architectures, where an
**initialization model** processes the first frame and a **forward model** processes all
subsequent frames while maintaining internal state between steps. It is designed for
dense point tracking across video sequences.

## How it works

Each `tick()` receives one or more GXF messages containing at least a `step` tensor
(int32, device memory) and a video frame:

1. **Step 0 (Init)** -- The Init TensorRT engine runs. Its outputs include state
   tensors that are copied into internal storage for the next step.
2. **Step > 0 (Forward)** -- The Forward TensorRT engine runs. State tensors from the
   previous step are fed as inputs, and updated state tensors are written back to
   internal storage after inference.

Query points are generated once at startup as an evenly-spaced grid and bound as the
`query_points` input on every tick.

## Supported model formats

- **ONNX** (`.onnx`) -- Automatically converted to a TensorRT engine on first use and
  cached in `engine_cache_dir`. Conversion can take several minutes.
- **Pre-built TensorRT engines** (`.engine` / `.plan`) -- Used directly, skipping
  conversion.

## Parameters

| Parameter                   | Type          | Default            | Description                                                             |
| --------------------------- | ------------- | ------------------ | ----------------------------------------------------------------------- |
| `model_file_path_init`      | `string`      | *(required)*       | Path to ONNX (or engine) model for initialization                       |
| `model_file_path_fwd`       | `string`      | *(required)*       | Path to ONNX (or engine) model for forward tracking                     |
| `engine_cache_dir`          | `string`      | *(required)*       | Directory for cached TensorRT engine files                              |
| `plugins_lib_namespace`     | `string`      | `""`               | TensorRT plugins library namespace                                      |
| `force_engine_update`       | `bool`        | `false`            | Force rebuild of TensorRT engines even if a cached engine exists        |
| `input_tensor_names_init`   | `string[]`    | *(required)*       | Input tensor names for the Init model                                   |
| `input_binding_names_init`  | `string[]`    | *(required)*       | Corresponding TensorRT binding names for Init inputs                    |
| `output_tensor_names_init`  | `string[]`    | *(required)*       | Output tensor names for the Init model                                  |
| `output_binding_names_init` | `string[]`    | *(required)*       | Corresponding TensorRT binding names for Init outputs                   |
| `input_tensor_names_fwd`    | `string[]`    | *(required)*       | Input tensor names for the Forward model                                |
| `input_binding_names_fwd`   | `string[]`    | *(required)*       | Corresponding TensorRT binding names for Forward inputs                 |
| `output_tensor_names_fwd`   | `string[]`    | *(required)*       | Output tensor names for the Forward model                               |
| `output_binding_names_fwd`  | `string[]`    | *(required)*       | Corresponding TensorRT binding names for Forward outputs                |
| `state_tensor_names`        | `string[]`    | *(required)*       | Tensor names treated as internal state (preserved across steps)         |
| `pool`                      | `Allocator`   | *(required)*       | GXF allocator instance for device tensor memory                         |
| `max_workspace_size`        | `int64`       | `67108864` (64 MB) | TensorRT builder max workspace size in bytes                            |
| `max_batch_size`            | `int32`       | `1`                | Max batch size for TensorRT optimization profiles                       |
| `enable_fp16`               | `bool`        | `false`            | Enable FP16 precision (ignored on TensorRT >= 10.13)                    |
| `relaxed_dimension_check`   | `bool`        | `true`             | Pad input rank with leading 1s when it is smaller than the binding rank |
| `verbose`                   | `bool`        | `false`            | Enable verbose TensorRT and codelet logging                             |
| `grid_size`                 | `int32`       | `15`               | Grid dimension N for query point generation (N x N points)              |
| `grid_height`               | `int32`       | `256`              | Image height used for query point grid spacing                          |
| `grid_width`                | `int32`       | `256`              | Image width used for query point grid spacing                           |
| `rx`                        | `Receiver[]`  | *(required)*       | List of input receivers                                                 |
| `tx`                        | `Transmitter` | *(required)*       | Output transmitter                                                      |

## Inputs

- **`step`** (`int32` tensor, device) -- Determines which model runs: `0` selects the
  Init model, any other value selects the Forward model.
- **`frame`** / **`video`** (`Tensor`, device) -- The video frame to process.
- Any additional tensors matching configured `input_tensor_names_*` entries are looked
  up by name from the incoming messages.

## Outputs

- All tensors listed in `output_tensor_names_*` for the selected model.
- **`step`** -- Passed through from the input.
- State tensors are also emitted as outputs but are additionally copied back into
  internal storage for the next tick.

## Query point generation

At startup the codelet generates a tensor of shape `(batch_size, grid_size * grid_size, 3)`
containing evenly-spaced query points with an 8-pixel margin from the image edges. Each
point is stored as `[t, x, y]` where `t = 0.0` (time index for the first frame), and
`x`, `y` are pixel coordinates derived from `grid_width` and `grid_height`.

## Requirements

- [Holoscan SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/overview.html)
- CUDA Runtime
- TensorRT (`nvinfer`, `nvinfer_plugin`)
- ONNX Parser (`nvonnxparser`)

## Building the extension

As part of Holohub, running CMake on Holohub and point to Holoscan SDK install tree.
