# Network Radar Pipeline
The Network Radar Pipeline demonstrates signal processing on data streamed via packets over a network. It showcases the use of both the Advanced Network Operator and Basic Network Operator to send or receive data, combined with the signal processing operators implemented in the Simple Radar Pipeline application.

The motivation for building this application is to demonstrate how data arrays can be assembled from packet data in real-time for low-latency, high-throughput sensor processing applications. The main components of this work are defining a message format and writing code connecting the network operators to the signal processing operators.

## Message format
The message format is defined by `RFPacket`. It is a byte array, represented by `RFPacket::payload`, where the first 16 bytes are reserved for metadata and the rest are used for representing complex I/Q samples. The metadata is:
- Sample index: The starting index for a single pulse/channel of the transmitted samples (2 bytes)
- Waveform ID: Index of the transmitted waveform (2 bytes)
- Channel index: Index of the channel (2 bytes)
- Pulse index: Index of the pulse (2 bytes)
- Number samples: Number of I/Q samples transmitted (2 bytes)
- End of array: Boolean - true if this is the last message for the waveform (2 bytes)

## Network Operator Connectors
See each operators' README before using / for more detailed information.
### Basic Network Operator Connector
Implementation in `basic_network_connectors`. Only supports CPU packet receipt / transmit. Uses cudaMemcpy to move data between network operator and MatX tensors.
### Advanced Network Operator Connector
Implementation in `advanced_network_connectors`. RX connector is only configured to run with GPUDirect enabled, in header-data split mode.
#### Testing RX on generic packet data
When using the Advanced network operator, the application supports testing the radar processing component in a "spoof packets" mode. This functionality allows for easier benchmarking of the application by ingesting generic packet data and writing in header fields such that the full radar pipeline will still be exercised. When "SPOOF_PACKET_DATA" (adv_networking_rx.h) is set to "true", the index of the packet will be used to set fields appropriately. This functionality is currently unsupported using the basic network operator connectors.

## Environment
Note: Dockerfile should be cross-compatible, but has only been tested on x86. Needs to be edited if different versions / architectures are required.

## Build
Please refer to the top level Holohub README.md file for information on how to build this application: `./run build network_radar_pipeline`.

## Run
Note: must properly configure YAML files before running.
- On Tx machine: `./build/applications/network_radar_pipeline/cpp/network_radar_pipeline source.yaml`
- On Rx machine: `./build/applications/network_radar_pipeline/cpp/network_radar_pipeline process.yaml`