# TapNext Inference Operator

The `tapnext_inference` operator runs TensorRT inference for TapNext-style architectures,
where an **initialization model** processes the first frame and a **forward model**
processes all subsequent frames while maintaining internal state between steps. It is
designed for dense point tracking across video sequences.

This operator wraps the
[`TapNextInference` GXF extension](../../gxf_extensions/tapnext_inference/README.md).

## `holoscan::ops::TapNextInferenceOp`

Operator class to perform TapNext inference (C++ and Python).

### How It Works

Each call receives one or more messages containing at least a `step` tensor (int32,
device memory) and a video frame:

1. **Step 0 (Init)** -- The Init TensorRT engine runs. Its outputs include state
   tensors that are copied into internal storage for the next step.
2. **Step > 0 (Forward)** -- The Forward TensorRT engine runs. State tensors from the
   previous step are fed as inputs, and updated state tensors are written back to
   internal storage after inference.

Query points are generated once at startup as an evenly-spaced grid and bound as the
`query_points` input on every tick.

### Supported Model Formats

- **ONNX** (`.onnx`) -- Automatically converted to a TensorRT engine on first use and
  cached in `engine_cache_dir`. Conversion can take several minutes.
- **Pre-built TensorRT engines** (`.engine` / `.plan`) -- Used directly, skipping
  conversion.

### Parameters

- **`model_file_path_init`**: Path to ONNX (or engine) model for initialization.
  - type: `std::string`
- **`model_file_path_fwd`**: Path to ONNX (or engine) model for forward tracking.
  - type: `std::string`
- **`engine_cache_dir`**: Directory for cached TensorRT engine files.
  - type: `std::string`
- **`plugins_lib_namespace`**: TensorRT plugins library namespace.
  - type: `std::string`
  - default: `""`
- **`force_engine_update`**: Force rebuild of TensorRT engines even if a cached engine exists.
  - type: `bool`
  - default: `false`
- **`input_tensor_names_init`**: Input tensor names for the Init model.
  - type: `std::vector<std::string>`
- **`input_binding_names_init`**: Corresponding TensorRT binding names for Init inputs.
  - type: `std::vector<std::string>`
- **`output_tensor_names_init`**: Output tensor names for the Init model.
  - type: `std::vector<std::string>`
- **`output_binding_names_init`**: Corresponding TensorRT binding names for Init outputs.
  - type: `std::vector<std::string>`
- **`input_tensor_names_fwd`**: Input tensor names for the Forward model.
  - type: `std::vector<std::string>`
- **`input_binding_names_fwd`**: Corresponding TensorRT binding names for Forward inputs.
  - type: `std::vector<std::string>`
- **`output_tensor_names_fwd`**: Output tensor names for the Forward model.
  - type: `std::vector<std::string>`
- **`output_binding_names_fwd`**: Corresponding TensorRT binding names for Forward outputs.
  - type: `std::vector<std::string>`
- **`state_tensor_names`**: Tensor names treated as internal state (preserved across steps).
  - type: `std::vector<std::string>`
- **`pool`**: Allocator instance for device tensor memory.
  - type: `std::shared_ptr<Allocator>`
- **`cuda_stream_pool`**: CUDA Stream Pool for asynchronous execution.
  - type: `std::shared_ptr<CudaStreamPool>`
- **`max_workspace_size`**: TensorRT builder max workspace size in bytes.
  - type: `int64_t`
  - default: `67108864` (64 MB)
- **`max_batch_size`**: Max batch size for TensorRT optimization profiles.
  - type: `int32_t`
  - default: `1`
- **`enable_fp16`**: Enable FP16 precision (ignored on TensorRT >= 10.13).
  - type: `bool`
  - default: `false`
- **`relaxed_dimension_check`**: Pad input rank with leading 1s when it is smaller than the binding rank.
  - type: `bool`
  - default: `true`
- **`verbose`**: Enable verbose TensorRT and operator logging.
  - type: `bool`
  - default: `false`
- **`grid_size`**: Grid dimension N for query point generation (N x N points).
  - type: `int32_t`
  - default: `15`
- **`grid_height`**: Image height used for query point grid spacing.
  - type: `int32_t`
  - default: `256`
- **`grid_width`**: Image width used for query point grid spacing.
  - type: `int32_t`
  - default: `256`

### Inputs

- **`receivers`** (`gxf::Entity`) -- One or more input messages containing:
  - **`step`** (`int32` tensor, device) -- `0` selects the Init model, any other value
    selects the Forward model.
  - **`frame`** / **`video`** (`Tensor`, device) -- The video frame to process.
  - Any additional tensors matching configured `input_tensor_names_*` entries.

### Outputs

- **`transmitter`** (`gxf::Entity`) -- Output message containing:
  - All tensors listed in `output_tensor_names_*` for the selected model.
  - **`step`** -- Passed through from the input.
  - State tensors are also emitted as outputs but are additionally copied back into
    internal storage for the next tick.

### Python API

The operator is available as `TapNextInferenceOp` via the Python bindings:

```python
from holohub.tapnext_inference import TapNextInferenceOp
```

The constructor accepts the same parameters as keyword arguments:

```python
tapnext = TapNextInferenceOp(
    self,
    model_file_path_init="/path/to/init.onnx",
    model_file_path_fwd="/path/to/fwd.onnx",
    engine_cache_dir="/tmp/engines",
    input_tensor_names_init=["video", "query_points"],
    input_binding_names_init=["video", "query_points"],
    output_tensor_names_init=["tracks", "visible", "state"],
    output_binding_names_init=["tracks", "visible", "state"],
    input_tensor_names_fwd=["video", "query_points", "state"],
    input_binding_names_fwd=["video", "query_points", "state"],
    output_tensor_names_fwd=["tracks", "visible", "state"],
    output_binding_names_fwd=["tracks", "visible", "state"],
    state_tensor_names=["state"],
    pool=allocator,
    cuda_stream_pool=cuda_stream_pool,
    name="tapnext_inference",
)
```
