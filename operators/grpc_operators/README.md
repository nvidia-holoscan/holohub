# Holohub gRPC Plugins for Holoscan SDK

## Overview

This directory contains the Holohub gRPC plugins for Holoscan SDK, including:

- `client`: gRPC client and Holoscan Operators for sending, receiving, and streaming data to a gRPC server.
- `server`: gRPC server and Holoscan Operators for handling requests from a gRPC client and transmitting data back to the client.
- `protos`: Protocol buffers definitions of Holoscan SDK.
- `common`: Tensor <-> protobuf converters and Holoscan Resources to handle incoming and outgoing data.

Please refer to the [gRPC h.264 Endoscopy Tool Tracking](../../applications/distributed/grpc/grpc_h264_endoscopy_tool_tracking/README.md) application for additional details.

The Python `GrpcService` permits plaintext only on loopback addresses and defaults
to `127.0.0.1`. Remote listeners require PEM `private_key`, `certificate_chain`,
and `root_certificates` (the trusted client CA) arguments to `initialize`, along
with a literal `host` IP address. Configured TLS always requires a trusted client
certificate. Python `EntityClientService` accepts a `credentials` argument from
`grpc.ssl_channel_credentials` for authenticated TLS connections and rejects
remote plaintext targets. See the [Python demo's mutual TLS setup](../../applications/distributed/grpc/grpc_endoscopy_tool_tracking/README.md#python-connections-between-hosts)
for configuration and testing instructions.
