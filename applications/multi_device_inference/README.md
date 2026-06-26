# Multi-Device Inference

Demonstrates the [`tensorrt_multi_device_inference`](../../operators/tensorrt_multi_device_inference)
operator: a single TensorRT engine **sharded across ≥2 GPUs** via TensorRT Multi-Device
(NCCL `DistCollective`). The pipeline is:

```
DeterministicTensorSourceOp  ->  TensorRtMultiDeviceInferenceOp (GPUs 0+1)  ->  ChecksumSinkOp
```

The source emits a deterministic FP32 tensor, the inference operator runs the sharded engine
across the configured GPUs, and the sink copies the result to host and prints a checksum.

## Requirements

- **TensorRT ≥ 11.0** (Multi-Device GA) and **NCCL** — note this is **not** in the stock Holoscan
  container (TRT 10); build/run in a TensorRT-11 + NCCL container.
- **≥ 2 homogeneous GPUs** (SM80+).
- Sharded engine plan(s) in `data/multi_device_inference/` (see below).

## Build engines

Provide either a single offline-sharded plan or per-rank weight-shard plans, e.g. a Megatron MLP
(`Y = (X·W1)·W2` with a trailing `AllReduce`, `W1` column-parallel and `W2` row-parallel). Place
them as `data/multi_device_inference/model.plan.rank0` and `...rank1` (matching `device_ids`), or a
single `model.plan` and list one path. The same approach validated the operator core on 2× B200.

## Run

```sh
./holohub run multi_device_inference --language cpp
# or directly:
<holohub_app_bin>/multi_device_inference <source>/multi_device_inference.yaml
```

Configure GPUs and engine paths in [`cpp/multi_device_inference.yaml`](cpp/multi_device_inference.yaml).

## Status

Experimental (`ranking: 3`). The Multi-Device inference core is validated on 2× B200
(TensorRT 11.1, max_rel 1.24e-05 vs the 1-GPU reference). The HoloHub build is exercised by CI,
which must provide a TensorRT-11 + NCCL container.
