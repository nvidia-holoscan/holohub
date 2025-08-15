
### Basic networking operator

The `basic_network_operator` operator provides a way to send and receive data over Linux sockets. The
destination can be on the same machine or over a network. The basic network operator contains separate
operators for transmit and receive. Users may choose one or the other, or use both in applications
requiring bidirectional traffic.

For TCP sockets the basic network operator only supports a single stream currently. Future versions
may expand this to launch multiple threads to listen on different streams.

The basic networking operators use class names: `BasicNetworkOpTx` and `BasicNetworkOpRx`

#### `nvidia::holoscan::basic_network_operator`

Basic networking operator

##### Receiver Configuration Parameters

- **`batch_size`**: Bytes in batch
  - type: `integer`
- **`max_payload_size`**: Maximum payload size for a single packet
  - type: `integer`
- **`udp_dst_port`**: UDP destination port for packets
  - type: `integer`
- **`l4_proto`**: Layer 4 protocol
  - type: `string` (`udp`/`tcp`)
- **`ip_addr`**: Destination IP address
  - type: `string`
- **`max_burst_interval` (Optional)**: Maximum time interval between bursts (ms)
  - type: `integer`

##### Transmitter Configuration Parameters

- **`max_payload_size`**: Maximum payload size for a single packet
  - type: `integer`
- **`udp_dst_port`**: UDP destination port for packets
  - type: `integer`
- **`l4_proto`**: Layer 4 protocol
  - type: `string` (`udp`/`tcp`)
- **`ip_addr`**: Destination IP address
  - type: `string`
- **`min_ipg_ns`**: Minimum inter-packet gap in nanoseconds
  - type: `integer`
- **`delete_payload` (Optional)**: Delete payload memory after sending (only applicable for C++ implementation)
  - type: `boolean`

##### Transmitter and Receiver Operator Parameters

The transmitter and receiver operator both use the `NetworkOpBurstParams` structure as input
and output to their ports, respectively. `NetworkOpBurstParams` contains the following fields:

- **`data`**: Pointer to batch of packet data
  - type: `uint8_t *`
- **`len`**: Length of total buffer in bytes
  - type: `integer`
- **`num_pkts`**: Number of packets in batch
  - type: `integer`

To receive messages from the Receive operator use the output port `burst_out`.
To send messages to the Transmit operator use the input port `burst_in`.