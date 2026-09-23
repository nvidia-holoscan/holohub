# TensorRT Multi-Device Inference Operator

Runs a **single TensorRT engine sharded across ≥2 GPUs** using TensorRT's **Multi-Device** feature
(NCCL `DistCollective` + `IExecutionContext::setCommunicator`), so one operator drives N GPUs.
Intended for models that do not fit — or run faster split — on a single GPU (tensor / Megatron
parallelism). TRT-28040.

> This is distinct from running independent models on different GPUs: here **one** model is split
> across GPUs and the partial results are combined in-engine via NCCL collectives.

## Requirements

- **TensorRT ≥ 11.0** (Multi-Device is GA in TensorRT 11) — **not** in the stock HoloHub/Holoscan
  container (TRT 10); build/run in a TensorRT-11 + NCCL container.
- **NCCL** (`libnccl`).
- **≥ 2 homogeneous GPUs** (SM80+) with peer access.
- Engine(s) sharded offline.

## Parameters

| Name | Type | Description |
| --- | --- | --- |
| `engine_paths` | `std::vector<std::string>` | TensorRT engine plan files, one per rank (index == rank). A **single** path = one offline-sharded plan deserialized on every rank; **N** paths = per-rank weight-shard plans (tensor parallelism). |
| `device_ids` | `std::vector<int32_t>` | Physical GPU id per rank; `device_ids[0]` is rank 0. Length = number of ranks (≥ 2). |
| `input_tensor_name` | `std::string` | Name of the input tensor on the incoming message (default `input`). |
| `output_tensor_name` | `std::string` | Name of the output tensor on the emitted message (default `output`). |
| `allocator` | `std::shared_ptr<Allocator>` | Allocator for the output tensor. |

## Ports

- **`in`** — `holoscan::gxf::Entity` carrying the FP32 input tensor named `input_tensor_name`.
- **`out`** — `holoscan::gxf::Entity` carrying the FP32 output tensor named `output_tensor_name`.

## How it works

`MultiDeviceTrt` (the reused, hardware-validated core): `ncclCommInitAll` → per-rank deserialize →
**concurrent** `setCommunicator` on every rank → host-bounce input replication to ranks 1..N-1 →
fan-out `enqueueV3` on all ranks → rank 0 produces the output. Rank 0 runs inside this operator;
ranks 1..N-1 are owned by `MultiDeviceTrt`.

## Building engines

Shard an ONNX/network into either one Multi-Device plan or per-rank weight-shard plans (e.g. with
`polygraphy` multi-device sharding, or a Megatron-style column/row split with an `AllReduce`).
See the companion `applications/multi_device_inference` for a runnable example and engine-build
notes.

## Status

Experimental (`ranking: 3`). The Multi-Device runtime core (`multidevice.cpp`) is validated on
2× NVIDIA B200 (TensorRT 11.1): a tensor-parallel MLP sharded across 2 GPUs matched the 1-GPU
reference (max_rel 1.24e-05). The HoloHub operator wrapper and sample application build is exercised
by HoloHub CI (which must provide a TensorRT-11 + NCCL container).
