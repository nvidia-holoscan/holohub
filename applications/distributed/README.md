# Distributed Applications

This directory contains applications designed to run in distributed environments, allowing them to offload heavy computation to a remote system or the cloud.

## UCX

Applications designed using Holoscan SDK's [Multi-Fragment](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_distributed_app.html) feature. Fragments communicate via [UCX](https://openucx.org/) (Unified Communication X).

- [ucx_endoscopy_tool_tracking](./ucx/ucx_endoscopy_tool_tracking/)
  A variation of the Endoscopy Tool Tracking application, separated into three fragments.

- [ucx_h264_endoscopy_tool_tracking](./ucx/ucx_h264_endoscopy_tool_tracking/)
  A variation of the h.264 Endoscopy Tool Tracking application, separated into three fragments.

## gRPC

Applications based on the [gRPC](https://grpc.io/) Server-Client concept. gRPC is a modern, open-source, high-performance Remote Procedure Call (RPC) framework. It enables efficient communication between services in and across data centers.

- [grpc_h264_endoscopy_tool_tracking](./grpc/grpc_h264_endoscopy_tool_tracking/)
  A h.264 Endoscopy Tool Tracking application that is separated into a gRPC server and a gRPC client.