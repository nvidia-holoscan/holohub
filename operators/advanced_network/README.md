### Advanced Network library

> [!NOTE]
> The Advanced Network library previously included standard operators for transmitting and receiving packets
> to/from the NIC, also referred to as the Advanced Network Operator (ANO). These operators were removed to
> lower overhead, as aggregation/disaggregation of the packets still needed to be done in separate operators.
> We plan to provide more full-fledged generic operators in the future. In the meantime, you can continue to use
> this library to develop Holoscan operators adapted to your use case, now including the direct packet transaction
> with the NIC. Referred to the [Benchmarking sample application](/applications/adv_networking_bench) for an example.

> [!WARNING]
> The library is undergoing large improvements as we aim to better support it as an NVIDIA product.
> API breakages might be more frequent until we reach version 1.0.

> [!TIP]
> Review the [High Performance Networking tutorial](/tutorials/high_performance_networking/README.md) for guided
> instructions to configure your system and test the Advanced Network library.

The Advanced Network library provides a way for users to achieve the highest throughput and lowest latency
for transmitting and receiving Ethernet frames out of and into Holoscan operators. Direct access to the NIC hardware
is available in userspace, thus bypassing the kernel's networking stack entirely.


#### Requirements

- Linux
- An NVIDIA NIC with a ConnectX-6 or later chip
- System tuning as described [here](/tutorials/high_performance_networking/README.md)
- DPDK 24.11.3 or higher
- MOFED 5.8-1.0.1.1 or later (included with DOCA package)
- MLNX5/IB drivers with peermem support - either through:
  - Inbox drivers (ubuntu kernel >= 5.4 and [< 6.8](https://discourse.ubuntu.com/t/nvidia-gpudirect-over-infiniband-migration-paths/44425))
  - NVIDIA optimized kernels (IGX OS, DGX BaseOS)
  - MLNX-OFED drivers, either from:
    - [DOCA-Host](https://developer.nvidia.com/doca-archive) 2.8 or later (install `mlnx-ofed-kernel-dkms` package or the `doca-ofed` meta-package for extra tooling)
    - _(deprecated)_ [MOFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/) 23.10 or later (`sudo ./mlnxofedinstall --kernel-only`)

> User-space libraries are including in the [Dockerfile](./Dockerfile) for each networking backend. Inspect this file if you wish to know what is needed to build and run on baremetal instead.

#### Features

- **High Throughput**: Hundreds of gigabits per second is possible with the proper hardware
- **Low Latency**: With direct access to the NIC's ring buffers, most latency incurred is only PCIe latency
  - Since the kernel's networking stack is bypassed, the user is responsible for defining the protocols used
		over the network. In most cases Ethernet, IP, and UDP are ideal for this type of processing because of their
		simplicity, but any type of protocol can be implemented or used. The advanced network library
		gives the option to use several primitives to remove the need for filling out these headers for basic packet types,
		but raw headers can also be constructed.
- **GPUDirect**: Optionally send data directly from the NIC to GPU, or directly from the GPU to NIC. GPUDirect has two modes:
  - Header-data split: Split the header portion of the packet to the CPU and the rest (payload) to the GPU. The split point is
    configurable by the user. This option should be the preferred method in most cases since it's easy to use and still
    gives near peak performance.
  - Batched GPU: Receive batches of whole packets directly into the GPU memory. This option requires the GPU kernel to inspect
    and determine how to handle packets. While performance may increase slightly over header-data split, this method
    requires more effort and should only be used for advanced users.
- **GPUComms**: Optionally control the send or receive communications from the GPU through the GPUDirect Async Kernel-Initiated network technology (enabled with the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) backend only).
- **Flow Configuration**: Configure the NIC's hardware flow engine for configurable patterns. Currently only UDP source
    and destination are supported.

#### Limitations

The limitations below will be removed in a future release.

- Only UDP fill mode is supported

#### Managers

Internally the advanced network library is implemented by different backends, each offering different features.

It is specified with the `manager` parameter, passed to the `advanced_network::adv_net_init` function before starting an application, along with all of the NIC parameters. This step allocates all packet buffers, initializes the queues on the NIC, and starts the appropriate number of internal threads to take packets off or put packets onto the NIC as fast as possible, using the backend-specific implementation.

Developers can then use the rest of the Advanced Network library API to send and receive packets, and do any
additional processing needed (e.g. aggregate, reorder, etc.), as described in the [API Structures](#api-structures) section.

> [!NOTE]
> To achieve zero copy throughout the whole pipeline only pointers are passed between each entity above. When the user
> receives the packets from the network library it's using the same buffers that the NIC wrote to either CPU or GPU
> memory. This architecture also implies that the user must explicitly decide when to free any buffers it's owning.
> Failure to free buffers will result in errors in the advanced network library not being able to allocate buffers.

##### DPDK

DPDK is an open-source userspace packet processing library supported across platforms and vendors.

It is the default manager, and can be set with the values `dpdk` or `default`.

Follow the instructions from the [adv_networking_bench](/applications/adv_networking_bench/README.md) README to build the operator and sample application with DPDK (default).

##### DOCA GPUNetIO

NVIDIA DOCA brings together a wide range of powerful APIs, libraries, and frameworks for programming and accelerating modern data center infrastructures​. [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) is one of the libraries included in the DOCA SDK. It enables the GPU to control, from a CUDA kernel, network communications directly interacting with the network card and completely removing the CPU from the critical data path.

If the application wants to enable GPU communications, it must chose `gpunetio` as backend. The behavior of the GPUNetIO backend is similar to the DPDK one except that the receive and send are executed by CUDA kernels. Specifically:

- Receive: a persistent CUDA kernel is running on a dedicated stream and keeps receiving packets, providing packets' info to the application level. Due to the nature of the operator, the CUDA receiver kernel now is responsible only to receive packets but in a real-world application, it can be extended to receive and process in real-time network packets (DPI, filtering, decrypting, byte modification, etc..) before forwarding packets to the application.
- Send: every time the application wants to send packets it launches one or more CUDA kernels to prepare data and create Ethernet packets and then (without the need of synchronizing) forward the send request to the operator. The operator then launches another CUDA kernel that in turn sends the packets (still no need to synchronize with the CPU). The whole pipeline is executed on the GPU. Due to the nature of the operator, the packets' creation and packets' send must be split in two CUDA kernels but in a real-word application, they can be merged into a single CUDA kernel responsible for both packet processing and packet sending.

Please refer to the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) programming guide to correctly configure your system before using this transport layer.

The GPUNetIO manager does not support the `split-boundary` option.

Follow the instructions from the [adv_networking_bench](/applications/adv_networking_bench/README.md) README to build the operator and sample application with DOCA GPUNetIO support.

##### RIVERMAX

NVIDIA Rivermax SDK
Optimized networking SDK for media and data streaming applications.
NVIDIA® Rivermax® offers a unique IP-based solution for any media and data streaming use case.
Rivermax together with NVIDIA GPU accelerated computing technologies unlocks innovation for a wide range of applications in Media and Entertainment (M&E), Broadcast, Healthcare, Smart Cities and more.
Rivermax leverages NVIDIA ConnectX® and BlueField® DPU hardware-streaming acceleration technology that enables direct data transfers to and from the GPU,
delivering best-in-class throughput and latency with minimal CPU utilization for streaming workloads.
Rivermax is the only fully-virtualized streaming solution that complies with the stringent timing and traffic flow requirements of the SMPTE ST 2110-21 specification.
Rivermax enables the future of cloud-based software-defined broadcasting.
Product release highlights, documentation, platform support, installation and usage guides can be found in the [Rivermax SDK Page](https://developer.nvidia.com/networking/rivermax-getting-started).
Frequently asked questions, customers product highlights, Video link and more are available on the [Rivermax Product Page](https://developer.nvidia.com/networking/rivermax).

To build and run the Dockerfile with `Rivermax` support, follow these steps:

- Visit the [Rivermax SDK Page](https://developer.nvidia.com/networking/rivermax-getting-started) to download the Rivermax Release SDK.
- Obtain a Rivermax developer license from the same page. This is necessary for using the SDK.
- Copy the downloaded SDK tar file (e.g., `rivermax_ubuntu2204_1.70.31.tar.gz`) into your current working directory.
  - You can adjust the path using the `RIVERMAX_SDK_ZIP_PATH` build argument if needed.
  - Modify the version using the `RIVERMAX_VERSION` build argument if you're using a different SDK version.
- Place the obtained Rivermax developer license file (`rivermax.lic`) into the `/opt/mellanox/rivermax/` directory.
- Build the operator and sample application with Rivermax support, see [adv_networking_bench](/applications/adv_networking_bench/README.md) README for instructions.

#### Configuration Parameters

##### Common Configuration

These common configurations are used by both TX and RX:

- **`version`**: Version of the config. Only 1 is valid currently.
  - type: `integer`
- **`master_core`**: Master core used to fork and join network threads. This core is not used for packet processing and can be
bound to a non-isolated core. Should differ from isolated cores in queues below.
  - type: `integer`
- **`manager`**: Backend networking library. default: `dpdk`. Other: `doca` (GPUNet IO), `rivermax`
  - type: `string`
- **`log_level`**: Backend log level. default: `warn`. Other: `trace` , `debug`, `info`, `error`, `critical`, `off`
  - type: `string`
- **`tx_meta_buffers`**: Metadata buffers for transmit. One buffer is used for each burst of packets (default: 4096)
  - type: `integer`
- **`rx_meta_buffers`**: Metadata buffers for receive. One buffer is used for each burst of packets (default: 4096)
  - type: `integer`

##### Memory regions

`memory_regions:` List of regions where buffers are stored.

- **`name`**: Memory Region name
  - type: `string`
- **`kind`**: Location. Best options are `device` (GPU), or `huge` (pages - CPU). Not recommended: `host` (CPU), `host_pinned` (CPU).
  - type: `string`
- **`affinity`**: GPU ID for GPU memory, NUMA Node ID for CPU memory
  - type: `integer`
- **`access`**: Permissions to the rdma memory region ( `local` or `rmda_read` or `rdma_write`)
  - type: `string`
- **`num_bufs`**: Higher value means more time to process, but less space on GPU BAR1.
Too low means risk of dropped packets from NIC having nowhere to write (Rx) or higher latency from buffering (Tx). Good rule of 👍 : 3x batch_size
  - type: `integer`
- **`buf_size`**: Size of buffer, equal to packet size or less if breaking down packets (ex: header data split)
  - type: `integer`

##### Interfaces
- **`interfaces`**:  List and configure ethernet interfaces
	full path: `cfg\interfaces\`
	- **`name`**: Name of the interfaca
	  - type: `string`
	- **`address`**: PCIe BDF address (lspci) or linux link name (ip link)
	  - type: `string`
	- **`rx|tx`** category of queues below
	full path: `cfg\interfaces\[rx|tx]`

##### Receive Configuration (rx)

- **`queues`**: List of queues on NIC
	type: `list`
	full path: `cfg\interfaces\rx\queues`
	- **`name`**: Name of queue
  		- type: `string`
	- **`id`**: Integer ID used for flow connection or lookup in operator compute method
  		- type: `integer`
	- **`cpu_core`**: CPU core ID. Should be isolated when CPU polls the NIC for best performance.. <mark>Not in use for Doca GPUNetIO</mark>
		Rivermax manager can accept coma separated list of CPU IDs
  		- type: `string`
	- **`batch_size`**: Number of packets in a batch passed from the NIC to the downstream operator. A
	larger number increases throughput but reduces end-to-end latency, as it takes longer to populate a single
	buffer. A smaller number reduces end-to-end latency but can also reduce throughput.
  		- type: `integer`
	- **`memory_regions`**: List of memory regions where buffers are stored. memory regions names are configured in the [Memory Regions](#memory-regions) section
		type: `list`
	- **`timeout_us`**: Timeout value that a batch will be sent on even if not enough packets to fill a batch were received
  		- type: `integer`
- **`flex_items`**: Flexible parser flow items
	type: `list`
	full path: `cfg\interfaces\rx\flex_items`
	- **`name`**: Name of flow item
  		- type: `string`	
	- **`id`**: ID of the flow item
  		- type: `integer`	
	- **`offset`**: Offset in bytes of where to match after the UDP header. Must be a multiple of 4 and < 28
  		- type: `integer`
	- **`udp_dst_port`**: UDP destination port for flex item match
			- type: `integer`

- **`flows`**: List of flows - rules to apply to packets, mostly to divert to the right queue. (<mark>Not in use for Rivermax manager</mark>)
  type: `list`
  full path: `cfg\interfaces\[rx|tx]\flows`
	- **`name`**: Name of the flow
	  - type: `string`
	- **`id`**: ID of the flow
	  - type: `integer`
	- **`action`**: Action section of flow (what happens. Currently only supports steering to a given queue)
	  - type: `sequence`
		- **`type`**: Type of action. Only `queue` is supported currently.
	  	-	 type: `string`
		- **`id`**: ID of queue to steer to
	  		- type: `integer`
	- **`match`**: Match section of flow
	  - type: `sequence`
		- **`udp_src`**: UDP source port or a range of ports (eg 1000-1010)
	  	- type: `integer`
		- **`udp_dst`**: UDP destination port or a range of ports (eg 1000-1010)
	  	- type: `integer`
		- **`ipv4_len`**: IPv4 payload length
	  	- type: `integer`
		- **`flex_item_id`**: Flex item ID from RX section. Flex items cannot be applied if UDP or IP matching above are used
	  	- type: `integer`
		- **`val`**: 32b value to match on
	  	- type: `integer`
		- **`mask`**: 32b mask to apply before the match
	  	- type: `integer`	
			

##### Extended Receive Configuration for Rivermax manager

- **`rivermax_rx_settings`**: Extended RX settings for Rivermax Manager. Rivermax Manager supports receiving the same stream from multiple redundant paths (IPO - Inline Packet Ordering).
	Each path is a combination of a source IP address, a destination IP address, a destination port, and a local IP address of the receiver device.
  type: `list`
  full path: `cfg\interfaces\rx\queues\rivermax_rx_settings`
	- **`memory_registration`**: Flag, when enabled, reduces the number of memory keys in use by registering all the memory in a single pass on the application side.
		<mark>Can be used only together with HDS enabled</mark>
  		- type: `boolean`
  		- default:`false`
	- **`max_path_diff_us`**: Sets the maximum number of microseconds that receiver waits for the same packet to arrive from a different stream (if IPO is enabled)
		- type: `integer`
		- default:`0`
	- **`ext_seq_num`**: The RTP sequence number is used by the hardware to determine the location of arriving packets in the receive buffer.
		The application supports two sequence number parsing modes: 16-bit RTP sequence number (default) and 32-bit extended sequence number,
		consisting of 16 low order RTP sequence number bits and 16 high order bits from the start of RTP payload. When set to `true` 32-bit ext. sequence number will be used
  		- type: `boolean`
  		- default:`true`
	- **`sleep_between_operations_us`**: Specifies the duration, in microseconds, that the receiver will pause or sleep between two consecutive receive (RX) operations.
  		- type: `integer`
  		- default:`0`
	- **`local_ip_addresses`**: List of Local NIC IP Addresses (one address per receiving path)
		- type: `sequence`
	- **`source_ip_addresses`**: List of Sender IP Addresses (one address per receiving path)
		- type: `sequence`
	- **`destination_ip_addresses`**: List of Destination IP Addresses (one address per receiving path), can be multicast
		- type: `sequence`
	- **`destination_ports`**: List of Destination IP ports (one port per receiving path)
		- type: `sequence`
	- **`stats_report_interval_ms`**: Specifies the duration, in milliseconds, that the receiver will display statistics in the log. Set `0` to disable statistics logging feature
  		- type: `integer`
  		- default:`0`
	- **`send_packet_ext_info`**: Enables the transmission of extended metadata for each received packet
  		- type: `boolean`
  		- default:`true`

- Example of the Rivermax queue configuration for redundant stream using HDS and GPU
  This example demonstrates receiving a redundant stream sent from a sender with source addresses 192.168.100.4 and 192.168.100.3.
  The stream is received via NIC which have local IP (same) 192.168.100.5 (listed twice, once per stream).
  The multicast addresses and UDP ports on which the stream is being received are 224.1.1.1:5001 and 224.1.1.2:5001
 The incoming packets are of size 1152 bytes. The initial 20 bytes are stripped from the payload as an  application header and placed in buffers allocated in RAM.
 The remaining 1132 bytes are placed in dedicated payload buffers.  In this case, the payload buffers are allocated in GPU 0 memory.
```YAML
    memory_regions:
    - name: "Data_RX_CPU"
      kind: "huge"
      affinity: 0
      access:
        - local
      num_bufs: 43200
      buf_size: 20
    - name: "Data_RX_GPU"
      kind: "device"
      affinity: 0
      access:
        - local
      num_bufs: 43200
      buf_size: 1132
    interfaces:
    - address: 0005:03:00.0
      name: data1
      rx:
        queues:
        - name: Data1
          id: 0
          cpu_core: '11'
          batch_size: 4320
          rivermax_rx_settings:
            settings_type: "ipo_receiver"
            memory_registration: true
            max_path_diff_us: 10000
            ext_seq_num: true
            sleep_between_operations_us: 0
            memory_regions:
            - "Data_RX_CPU"
            - "Data_RX_GPU"
            local_ip_addresses:
            - 192.168.100.5
            - 192.168.100.5
            source_ip_addresses:
            - 192.168.100.4
            - 192.168.100.4
            destination_ip_addresses:
            - 224.1.1.1
            - 224.1.1.2
            destination_ports:
            - 50001
            - 50001
            stats_report_interval_ms: 3000
            send_packet_ext_info: true

```

##### Transmit Configuration (tx)
 
- **`queues`**: List of queues on NIC
	type: `list`
	full path: `cfg\interfaces\tx\queues`
	- **`name`**: Name of queue
  		- type: `string`
	- **`id`**: Integer ID used for flow connection or lookup in operator compute method
  		- type: `integer`
	- **`cpu_core`**: CPU core ID. Should be isolated when CPU polls the NIC for best performance.. <mark>Not in use for Doca GPUNetIO</mark>
		Rivermax manager can accept coma separated list of CPU IDs
  		- type: `string`
	- **`batch_size`**: Number of packets in a batch that the NIC needs to receive from the upstream operator before
	sending them over the network. A larger number increases throughput but reduces end-to-end latency.
	A smaller number reduces end-to-end latency but can also reduce throughput.
  		- type: `integer`
	- **`memory_regions`**: List of memory regions where buffers are stored. memory regions names are configured in the [Memory Regions](#memory-regions) section
		type: `list`
	- **`accurate_send`**: Accurate TX sending enabled for sending packets at a specific PTP timestamp
  		- type: `boolean`

##### Transmit Configuration (tx)

- **`queues`**: List of queues on NIC
  **Type**: `list`
  **Full Path**: `cfg\interfaces\tx\queues`

  - **`name`**: Name of the queue
    - **Type**: `string`

  - **`id`**: Integer ID used for flow connection or lookup in operator compute method
    - **Type**: `integer`

  - **`cpu_core`**: CPU core ID. Should be isolated when CPU polls the NIC for best performance. <mark>Not in use for DOCA GPUNetIO</mark>. Rivermax Manager can accept comma-separated list of CPU IDs.
    - **Type**: `string`

  - **`batch_size`**: Number of packets in batch passed between operators. Larger values increase throughput at cost of latency.
    - **Type**: `integer`

  - **`memory_regions`**: List of memory regions where buffers are stored (configured in Memory Regions section)
    - **Type**: `list`
##### Extended Transmit Configuration for Rivermax manager
The Rivermax TX configuration enables hardware-assisted SMPTE 2110-20 compliant video streaming with:
- Precision timestamping via PCIe PTP clock synchronization
- Jitter-free packetization of rasterized video frames
- Automatic UDP checksum offload
- Traffic shaping for constant bitrate delivery

  - **`rivermax_tx_settings`**: Extended TX settings for SMPTE 2110-20 media streaming
    **Type**: `sequence`
    **Full Path**: `cfg\interfaces\tx\queues\rivermax_tx_settings`

    - **`settings_type`**: Transmission mode. Must be `media_sender` for video
      - **Type**: `string`
      - **Required**: Yes
      - **Valid Values**: `media_sender`

    - **`memory_registration`**: Enables bulk registration of GPU/CPU memory regions with NIC for zero-copy transfers. Recommended for high-throughput scenarios.
      - **Type**: `boolean`
      - **Default**: `true`

    - **`memory_allocation`**:  I\O Memory allocated by application
      - **Type**: `boolean`
      - **Default**: `true`

    - **`memory_pool_location`**: Buffer memory type (`device/huge_pages/host_pinned/host`)
      - **Type**: `string`
      - **Required**: Yes

    - **`local_ip_address`**: Source IP address bound to transmitting network interface.
      - **Type**: `string`
      - **Required**: Yes

    - **`destination_ip_address`**: Unicast/Multicast group address for media stream distribution.
      - **Type**: `string`
      - **Required**: Yes

    - **`destination_port`**: UDP port number for media stream transmission
      - **Type**: `integer`
      - **Range**: 1024-65535
      - **Required**: Yes

    - **`video_format`**: Defines pixel sampling structure per SMPTE ST 2110-20
      - **Type**: `string`
      - **Required**: Yes
      - **Valid Values**: `YCbCr-4:2:2`, `YCbCr-4:4:4`,`YCbCr-4:2:0`, `RGB`

    - **`bit_depth`**: Color component quantization precision
      - **Type**: `integer`
      - **Required**: Yes
      - **Valid Values**: `8`, `10`, `12`

    - **`frame_width`**: Horizontal resolution in pixels
      - **Type**: `integer`
      - **Required**: Yes
      -  **Valid Values**: `1920` for HD, `3840` for 4K UHD

    - **`frame_height`**: The vertical resolution of the video in pixels
      - **Type**: `integer`
      - **Required**: Yes
      -  **Valid Values**: `1080` for HD, `2160` for 4K UHD

    - **`frame_rate`**: Frame rate in fps
      - **Type**: `integer`
      - **Required**: Yes
      - **Valid Values**:  `24`, `25`, `30`, `50`, and `60`

    - **`dummy_sender`**: Test mode without NIC transmission
      - **Type**: `boolean`
      - **Default**: `false`

    - **`stats_report_interval_ms`**: Transmission stats logging interval (0=disable)
      - **Type**: `integer`
      - **Default**: `0`

    - **`verbose`**: Enable detailed transmission logging
      - **Type**: `boolean`
      - **Default**: `false`

    - **`sleep_between_operations`**: Add inter-burst delays for timing sync
      - **Type**: `boolean`
      - **Default**: `false`

#### Example Configuration
```YAML
    memory_regions:
    - name: "Data_TX_CPU"
      kind: "huge"
      affinity: 0
      num_bufs: 43200
      buf_size: 20
    - name: "Data_TX_GPU"
      kind: "device"
      affinity: 0
      num_bufs: 43200
      buf_size: 1200

    interfaces:
    - name: "tx_port"
      address: cc:00.1
      tx:
        queues:
        - name: "tx_q_1"
          id: 0
          cpu_core:  "13"
          batch_size: 4320
          output_port: "bench_tx_out_1"
          memory_regions:
          - "Data_TX_CPU"
          - "Data_TX_GPU"
          rivermax_tx_settings:
            settings_type: "media_sender"
            memory_registration: true
            memory_allocation: true
            memory_pool_location: "host_pinned"
            #allocator_type: "huge_page_2mb"
            verbose: true
            sleep_between_operations: false
            local_ip_address: 2.1.0.12
            destination_ip_address: 224.1.1.2
            destination_port: 50001
            stats_report_interval_ms: 1000
            send_packet_ext_info: true
            video_format: YCbCr-4:2:2
            bit_depth: 10
            frame_width: 1920
            frame_height: 1080
            frame_rate: 60
            dummy_sender: false

```

#### API Structures

The Advanced Network library uses a common structure named `BurstParams` to pass data to/from other operators. `BurstParams` provides pointers to packet memory locations (e.g., CPU or GPU) and contains metadata needed by any operator to track allocations. Interacting with `BurstParams` should only be done with the helper functions described below.

#### Example API Usage

For an entire list of API functions, please see the `advanced_network/common.h` header file.

##### Receive

The section below describes a workflow using GPUDirect to receive packets using header-data split. The job of the user's operator(s)
is to process and free the buffers as quickly as possible. This might be copying to interim buffers or freeing before the entire
pipeline is done processing. This allows the networking piece to use relatively few buffers while still achieving very high rates.

The first step in receiving from the NIC is to receive a `BurstParams` structure when a batch is complete:

```cpp
BurstParams *burst;
int port_id_ = 0;
int queue_id_ = 0;
auto status = get_rx_burst(&burst, port_id_, queue_id_);
```

The packets arrive in scattered packet buffers. Depending on the application, you may need to iterate through the packets to
aggregate them into a single buffer. Alternatively the operator handling the packet data can operate on a list of packet
pointers rather than a contiguous buffer. Below is an example of aggregating separate GPU packet buffers into a single GPU
buffer:

```cpp
  for (int p = 0; p < get_num_packets(burst); p++) {
    h_dev_ptrs_[aggr_pkts_recv_ + p]   = get_cpu_packet_ptr(burst, p);
    ttl_bytes_in_cur_batch_           += get_gpu_packet_length(burst, p) + sizeof(UDPPkt);
  }

  simple_packet_reorder(buffer, h_dev_ptrs, packet_len, burst->hdr.num_pkts);
```

For this example we are tossing the header portion (CPU), so we don't need to examine the packets. Since we launched a reorder
kernel to aggregate the packets in GPU memory, we are also done with the GPU pointers. All buffers may be freed for the NIC to reuse at this point:

```cpp
free_all_burst_packets_and_burst(burst_bufs_[b]);
```

##### Transmit

Transmitting packets works similar to the receive side, except the user is tasked with filling out the packets as much as it
needs to. As mentioned above, helper functions are available to fill in most boilerplate header information if that doesn't
change often.

Before sending packets, the user's transmit operator must request a buffer from the NIC:

```cpp
auto burst = create_tx_burst_params();
set_header(burst, port_id, queue_id, batch_size, num_segments);
if ((ret = get_tx_packet_burst(burst)) != Status::SUCCESS) {
  HOLOSCAN_LOG_ERROR("Error returned from get_tx_packet_burst: {}", static_cast<int>(ret));
  return;
}
```

The code above creates a shared `BurstParams`, and uses `get_tx_packet_burst` to populate the burst buffers with valid packet buffers. On success, the buffers inside the burst structure will be allocated and are ready to be filled in. Each packet must be filled in by the user. In this example we loop through each packet and populate a buffer:

```cpp
for (int num_pkt = 0; num_pkt < get_num_packets(burst); num_pkt++) {
  void *payload_src = data_buf + num_pkt * payload_size;
  if (set_udp_payload(burst, num_pkt, payload_src, payload_size) != Status::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create packet {}", num_pkt);
  }
}
```

The code iterates over the number of packets in the burst (defined above by the user) and passes a pointer
to the payload and the packet size to `set_udp_payload`. In this example our configuration is using `fill_mode`
"udp" on the transmitter, so `set_udp_payload` will populate the Ethernet, IP, and UDP headers. The payload
pointer passed by the user is also copied into the buffer. Alternatively a user could use the packet buffers
directly as output from a previous stage to avoid this extra copy.

With the `BurstParams` populated, the burst can be sent off to the NIC:

```cpp
send_tx_burst(burst);
```