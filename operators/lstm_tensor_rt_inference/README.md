### Custom LSTM Inference

The `lstm_tensor_rt_inference` extension provides LSTM (Long-Short Term Memory) stateful inference module using TensorRT.

#### `nvidia::holoscan::lstm_tensor_rt_inference::TensorRtInference`

Codelet, taking input tensors and feeding them into TensorRT for LSTM inference.

This implementation is based on `nvidia::gxf::TensorRtInference`.
`input_state_tensor_names` and `output_state_tensor_names` parameters are added to specify tensor names for states in LSTM model.

##### Parameters

- **`model_file_path`**: Path to ONNX model to be loaded
  - type: `std::string`
- **`engine_cache_dir`**: Path to a directory containing cached generated engines to be serialized and loaded from
  - type: `std::string`
- **`plugins_lib_namespace`**: Namespace used to register all the plugins in this library (default: `""`)
  - type: `std::string`
- **`force_engine_update`**: Always update engine regard less of existing engine file. Such conversion may take minutes (default: `false`)
  - type: `bool`
- **`input_tensor_names`**: Names of input tensors in the order to be fed into the model
  - type: `std::vector<std::string>`
- **`input_state_tensor_names`**: Names of input state tensors that are used internally by TensorRT
  - type: `std::vector<std::string>`
- **`input_binding_names`**: Names of input bindings as in the model in the same order of what is provided in input_tensor_names
  - type: `std::vector<std::string>`
- **`output_tensor_names`**: Names of output tensors in the order to be retrieved from the model
  - type: `std::vector<std::string>`
- **`input_state_tensor_names`**: Names of output state tensors that are used internally by TensorRT
  - type: `std::vector<std::string>`
- **`output_binding_names`**: Names of output bindings in the model in the same order of of what is provided in output_tensor_names
  - type: `std::vector<std::string>`
- **`pool`**: Allocator instance for output tensors
  - type: `gxf::Handle<gxf::Allocator>`
- **`cuda_stream_pool`**: Instance of gxf::CudaStreamPool to allocate CUDA stream
  - type: `gxf::Handle<gxf::CudaStreamPool>`
- **`max_workspace_size`**: Size of working space in bytes (default: `67108864l` (64MB))
  - type: `int64_t`
- **`dla_core`**: DLA Core to use. Fallback to GPU is always enabled. Default to use GPU only (`optional`)
  - type: `int32_t`
- **`max_batch_size`**: Maximum possible batch size in case the first dimension is dynamic and used as batch size (default: `1`)
  - type: `int32_t`
- **`enable_fp16_`**: Enable inference with FP16 and FP32 fallback (default: `false`)
  - type: `bool`
- **`verbose`**: Enable verbose logging on console (default: `false`)
  - type: `bool`
- **`relaxed_dimension_check`**: Ignore dimensions of 1 for input tensor dimension check (default: `true`)
  - type: `bool`
- **`rx`**: List of receivers to take input tensors
  - type: `std::vector<gxf::Handle<gxf::Receiver>>`
- **`tx`**: Transmitter to publish output tensors
  - type: `gxf::Handle<gxf::Transmitter>`
