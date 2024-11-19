### Advanced Network Operator

The Advanced Network Operator provides a way for users to achieve the highest throughput and lowest latency
for transmitting and receiving Ethernet frames out of and into their operators. Direct access to the NIC hardware
is available in userspace using this operator, thus bypassing the kernel's networking stack entirely. With a
properly tuned system the advanced network operator can achieve hundreds of Gbps with latencies in the low
microseconds. Performance is highly dependent on system tuning, packet sizes, batch sizes, and other factors.
The data may optionally be sent to the GPU using GPUDirect to prevent extra copies to and from the CPU.

Since the kernel's networking stack is bypassed, the user is responsible for defining the protocols used
over the network. In most cases Ethernet, IP, and UDP are ideal for this type of processing because of their
simplicity, but any type of protocol can be implemented or used. The advanced network operator
gives the option to use several primitives to remove the need for filling out these headers for basic packet types,
but raw headers can also be constructed.

#### Requirements

- Linux
- An NVIDIA NIC with a ConnectX-6 or later chip
- System tuning as described below
- DPDK 22.11
- MOFED 5.8-1.0.1.1 or later
- DOCA 2.7 or later

#### Features

- **High Throughput**: Hundreds of gigabits per second is possible with the proper hardware
- **Low Latency**: With direct access to the NIC's ring buffers, most latency incurred is only PCIe latency
- **GPUDirect**: Optionally send data directly from the NIC to GPU, or directly from the GPU to NIC. GPUDirect has two modes:
  - Header-data split: Split the header portion of the packet to the CPU and the rest (payload) to the GPU. The split point is
    configurable by the user. This option should be the preferred method in most cases since it's easy to use and still
    gives near peak performance.
  - Batched GPU: Receive batches of whole packets directly into the GPU memory. This option requires the GPU kernel to inspect
    and determine how to handle packets. While performance may increase slightly over header-data split, this method
    requires more effort and should only be used for advanced users.
- **GPUComms**: Optionally control the send or receive communications from the GPU through the GPUDirect Async Kernel-Initiated network technology (enabled with the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) transport layer only).
- **Flow Configuration**: Configure the NIC's hardware flow engine for configurable patterns. Currently only UDP source
    and destination are supported.

#### Limitations

The limitations below will be removed in a future release.

- Only UDP fill mode is supported

#### Implementation

Internally the advanced network operator can be implemented by different transport layers, each offering different features.
The network transport layer must be specified at the beginning of the application using the ANO API.

##### DPDK

DPDK is an open-source userspace packet processing library supported across platforms and vendors.
While the DPDK interface is abstracted away from users of the advanced network operator,
the method in which DPDK integrates with Holoscan is important for understanding how to achieve the highest performance and for debugging.

When the advanced network operator is compiled/linked against a Holoscan application, an instance of the DPDK manager
is created, waiting to accept configuration. When either an RX or TX advanced network operator is defined in a
Holoscan application, their configuration is sent to the DPDK manager. Once all advanced network operators have initialized,
the DPDK manager is told to initialize DPDK. At this point the NIC is configured using all parameters given by the operators.
This step allocates all packet buffers, initializes the queues on the NIC, and starts the appropriate number of internal
threads. The job of the internal threads is to take packets off or put packets onto the NIC as fast as possible. They
act as a proxy between the advanced network operators and DPDK by handling packets faster than the operators may be
able to.

To achieve zero copy throughout the whole pipeline only pointers are passed between each entity above. When the user
receives the packets from the network operator it's using the same buffers that the NIC wrote to either CPU or GPU
memory. This architecture also implies that the user must explicitly decide when to free any buffers it's owning.
Failure to free buffers will result in errors in the advanced network operators not being able to allocate buffers.


##### DOCA

NVIDIA DOCA brings together a wide range of powerful APIs, libraries, and frameworks for programming and accelerating modern data center infrastructures​. The DOCA SDK composed by a variety of C/C++ API for different purposes​, exposing all the features supported by NVIDIA hardware and platforms. [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) is one of the libraries included in the SDK and it enables the GPU to control, from a CUDA kernel, network communications directly interacting with the network card and completely removing the CPU from the critical data path.

If the application wants to enable GPU communications, it must chose DOCA as transport layer. The behaviour of the DOCA transport layer is similar to the DPDK one except that the receive and send are executed by CUDA kernels. Specifically:
- Receive: a persistent CUDA kernel is running on a dedicated stream and keeps receiving packets, providing packets' info to the application level. Due to the nature of the operator, the CUDA receiver kernel now is responsible only to receive packets but in a real-world application, it can be extended to receive and process in real-time network packets (DPI, filtering, decrypting, byte modification, etc..) before forwarding packets to the application.
- Send: every time the application wants to send packets it launches one or more CUDA kernels to prepare data and create Ethernet packets and then (without the need of synchronizing) forward the send request to the operator. The operator then launches another CUDA kernel that in turn sends the packets (still no need to synchronize with the CPU). The whole pipeline is executed on the GPU. Due to the nature of the operator, the packets' creation and packets' send must be split in two CUDA kernels but in a real-word application, they can be merged into a single CUDA kernel responsible for both packet processing and packet sending.

Please refer to the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) programming guide to correctly configure your system before using this transport layer.

DOCA transport layer doesn't support the `split-boundary` option.

To build and run the ANO Dockerfile with DOCA support, please follow the steps below:

```
# To build Docker image
./dev_container build --docker_file operators/advanced_network/Dockerfile --img holohub-doca:doca-28-ubuntu2204 --no-cache

# Launch DOCA container
./operators/advanced_network/run_doca.sh

# To build operator + app from main dir
./run build adv_networking_bench --configure-args "-DANO_MGR=doca"

# Run app
./build/adv_networking_bench/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_doca_tx_rx.yaml
```

<mark>Receiver side, CUDA Persistent kernel note</mark>
To get the best performance on the receive side, the Advanced Network Operator must be built with with `RX_PERSISTENT_ENABLED` set to 1 which enables the CUDA receiver kernel to run persistently for the whole execution. For Holoscan internal reasons (not related to the DOCA library), a persistent CUDA kernel may cause issues on some applications on the receive side. This issue is still under investigation.
If this happens, there are two options:
- build the Advanced Network Operator with `RX_PERSISTENT_ENABLED` set to 0
- keep the `RX_PERSISTENT_ENABLED` set to 1 and enable also MPS setting `MPS_ENABLED` to 1. Then, MPS should be enabled on the system:
```
export CUDA_MPS_PIPE_DIRECTORY=/var
export CUDA_MPS_LOG_DIRECTORY=/var
sudo -E nvidia-cuda-mps-control -d
sudo -E echo start_server -uid 0 | sudo -E nvidia-cuda-mps-control
```

This should solve all problems. Both `RX_PERSISTENT_ENABLED` and `MPS_ENABLED` are defined in `operators/advanced_network/managers/doca/adv_network_doca_mgr.h`.
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
While the Rivermax interface is abstracted away from users of the advanced network operator, the method in which Rivermax integrates with Holoscan
is important for understanding how to achieve the highest performance and for debugging.

When the advanced network operator is compiled/linked against a Holoscan application, an instance of the Rivermax manager is created, waiting to accept configuration. 
When either an RX or TX advanced network operator is defined in aHoloscan application, their configuration is sent to the Rivermax manager.
Once all advanced network operators have initialized, the Rivermax manager is told to initialize Rivermax. At this point the NIC is configured using all parameters given by the operators.
This step allocates all packet buffers, initializes the queues on the NIC, and starts the appropriate number of internal threads. 
The job of the internal threads is to take packets off or put packets onto the NIC as fast as possible. 
They act as a proxy between the advanced network operators and Rivermax by handling packets faster than the operators may be able to.

To achieve zero copy throughout the whole pipeline only pointers are passed between each entity above. 
When the user receives the packets from the network operator it's using the same buffers that the NIC wrote to either CPU or GPU memory.
This architecture also implies that the user must explicitly decide when to free any buffers it's owning.
Failure to free buffers will result in errors in the advanced network operators not being able to allocate buffers.
Rivermax manager supports receiving the same stream from multiple redundant paths.
Each path is a combination of a source IP address, a destination IPaddress, a destination port, and a local IP address of the receiver device.
Single path receive supports packet reordering within the NIC, multi-pathreceive also adds recovery of missing packets from other streams.

To build and run the ANO Dockerfile with `Rivermax` support, follow these steps:

- Visit the [Rivermax SDK Page](https://developer.nvidia.com/networking/rivermax-getting-started) to download the Rivermax Release SDK.
- Obtain a Rivermax developer license from the same page. This is necessary for using the SDK.
- Copy the downloaded SDK tar file (e.g., `rivermax_ubuntu2204_1.60.1.tar.gz`) into your current working directory.
  - You can adjust the path using the `RIVERMAX_SDK_ZIP_PATH` build argument if needed.
  - Modify the version using the `RIVERMAX_VERSION` build argument if you're using a different SDK version.
- Place the obtained Rivermax developer license file (`rivermax.lic`) into the `/opt/mellanox/rivermax/` directory. You can change this path in the run_rivermax.sh script if necessary
- Build the Docker image:

```
./dev_container build --docker_file operators/advanced_network/Dockerfile --img holohub:rivermax --build-args "--target rivermax"
```

- Launch Rivermax container

```
# Launch Rivermax container
./operators/advanced_network/run_rivermax.sh

# To build operator + app from main dir
./run build adv_networking_bench --configure-args "-DANO_MGR=rivermax"

# Run app
./build/adv_networking_bench/applications/adv_networking_bench/cpp/adv_networking_bench  adv_networking_bench_rmax_rx.yaml
```



#### System Tuning

From a high level, tuning the system for a low latency workload prevents latency spikes large enough to cause anomalies
in the application. This section details how to perform the basic tuning steps needed on both a Clara AGX and Orin IGX systems.

##### Create Hugepages

Hugepages give the kernel access to a larger page size than the default (usually 4K) which reduces the number of memory
translations that have to be actively maintained in MMUs. 1GB hugepages are ideal, but 2MB may be used as well if 1GB is not
available. To configure 1GB hugepages:

```
sudo mkdir /mnt/huge
sudo mount -t hugetlbfs nodev /mnt/huge
sudo sh -c "echo nodev /mnt/huge hugetlbfs pagesize=1GB 0 0 >> /etc/fstab"
```

##### Linux Boot Command Line

The Linux boot command line allows configuration to be injected into Linux before booting. Some configuration options are
only available at the boot command since they must be provided before the kernel has started. On the Orin IGX
editing the boot command can be done with the following configuration:

```
sudo vim /etc/default/grub
# Find the line starting with APPEND and add the following

isolcpus=6-11 nohz_full=6-11 irqaffinity=0-5 rcu_nocbs=6-11 rcu_nocb_poll tsc=reliable audit=0 nosoftlockup default_hugepagesz=1G hugepagesz=1G hugepages=2
```

The settings above isolate CPU cores 6-11 on the Orin and 4-7 on the Clara, and turn 1GB hugepages on.

For non-IGX or AGX systems please look at the documentation for your system to change the boot command.

##### Setting the CPU governor

The CPU governor reduces power consumption by decreasing the clock frequency of the CPU when cores are idle. While this is useful
in most environments, increasing the clocks from an idle period can cause long latency stalls. To disable frequency scaling:

```
sudo apt install cpufrequtils
sudo sed -i 's/^GOVERNOR=.*/GOVERNOR="performance"/' /etc/init.d/cpufrequtils
```

Reboot the system after these changes.

##### Permissions

DPDK typically requires running as a root user. If you wish to run as a non-root user, you may follow the directions here:
http://doc.dpdk.org/guides/linux_gsg/enable_func.html

If running in a container, you will need to run in privileged container, and mount your hugepages mount point from above into the container. This
can be done as part of the `docker run` command by adding the following flags:

```
-v /mnt/huge:/mnt/huge \
--privileged \
```    

#### Configuration Parameters

The advanced network operator contains a separate operator for both transmit and receive. This allows applications to choose
whether they need to handle bidirectional traffic or only unidirectional. Transmit and receive are configured separately in
a YAML file, and a common configuration contains items used by both directions. Each configuration section is described below.

##### Common Configuration

The common configuration container parameters are used by both TX and RX:

- **`version`**: Version of the config. Only 1 is valid currently.
  - type: `integer`
- **`master_core`**: Master core used to fork and join network threads. This core is not used for packet processing and can be
bound to a non-isolated core. Should differ from isolated cores in queues below.
  - type: `integer`
- **`manager`**: Backend networking library. default: `dpdk`. Other: `doca` (GPUNet IO), `rivermax`
  - type: `string`
- **`log_level`**: Backend log level. default: `warn`. Other: `trace` , `debug`, `info`, `error`, `critical`, `off`
  - type: `string`

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
	- **`batch_size`**: Number of packets in a batch that is passed between the advanced network operator and the user's operator. A
	larger number increases throughput and latency by requiring fewer messages between operators, but takes longer to populate a single
	buffer. A smaller number reduces latency and bandwidth by passing more messages.
  		- type: `integer`
	- **`split_boundary`**: HDS (Header Data Split) Split point in bytes between header and payload. If set to 0 HDS is disabled
  		- type: `integer`
	- **`output_port`**:  Name of the ANO Rx operator output port for aggregator operators to connect to
  		- type: `string`
	- **`memory_regions`**: List of memory regions where buffers are stored. memory regions names are configured in the [Memory Regions](#memory-regions) section
		type: `list`

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
		- **`udp_src`**: UDP source port
	  	- type: `integer`
		- **`udp_dst`**: UDP destination port
	  	- type: `integer`
		- **`ipv4_len`**: IPv4 payload length
	  	- type: `integer`      

##### Extended Receive Configuration for Rivermax manager
- **`rmax_rx_settings`**: Extended RX settings for Rivermax Manager. Rivermax Manager supports receiving the same stream from multiple redundant paths (IPO - Inline Packet Ordering).
	Each path is a combination of a source IP address, a destination IP address, a destination port, and a local IP address of the receiver device.
  type: `list`
  full path: `cfg\interfaces\rx\queues\rmax_rx_settings`
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
	- **`rx_stats_period_report_ms`**: Specifies the duration, in milliseconds, that the receiver will display statistics in the log. Set `0` to disable statistics logging feature
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
          id: 1
          cpu_core: '11'
          batch_size: 4320
          output_port: bench_rx_out_1    
          rmax_rx_settings:
            memory_registration: true
            max_path_diff_us: 100
            ext_seq_num: true
            sleep_between_operations_us: 100
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
            rx_stats_period_report_ms: 3000
            send_packet_ext_info: true
          
```

##### Transmit Configuration (tx)
 (<mark>Current version of Rivermax manager doesn't support TX</mark>)

- **`queues`**: List of queues on NIC
	type: `list`
	full path: `cfg\interfaces\tx\queues`
	- **`name`**: Name of queue
  		- type: `string`
 	- **`id`**: Integer ID used for flow connection or lookup in operator compute method
  		- type: `integer`
	- **`cpu_core`**: CPU core ID. Should be isolated when CPU polls the NIC for best performance.. <mark>Not in use for Doca GPUNet IORivermax manager</mark>
		Rivermax manager can accept coma separated list of CPU IDs
  		- type: `string`
	- **`batch_size`**: Number of packets in a batch that is passed between the advanced network operator and the user's operator. A
	larger number increases throughput and latency by requiring fewer messages between operators, but takes longer to populate a single
	buffer. A smaller number reduces latency and bandwidth by passing more messages.
  		- type: `integer`
	- **`split_boundary`**: HDS (Header Data Split) Split point in bytes between header and payload. If set to 0 HDS is disabled
  		- type: `integer`
	- **`memory_regions`**: List of memory regions where buffers are stored. memory regions names are configured in the [Memory Regions](#memory-regions) section
		type: `list`


#### API Structures

Both the transmit and receive operators use a common structure named `AdvNetBurstParams` to pass data to/from other operators. `AdvNetBurstParams` provides pointers to packet memory locations (e.g., CPU or GPU) and contains metadata needed by the operator to track allocations. Since the advanced network operator utilizes a generic interface that does not expose the underlying low-level network card library, interacting with `AdvNetBurstParams` is mostly done with the helper functions described below. A user should never modify any members of `AdvNetBurstParams` directly, as this may break in future versions. The `AdvNetBurstParams` is described below:

```
struct AdvNetBurstParams {
  AdvNetBurstHdr hdr;

  std::array<void**, MAX_NUM_SEGS> pkts;
  std::array<uint32_t*, MAX_NUM_SEGS> pkt_lens;
  void** pkt_extra_info;
  cudaEvent_t event;
};
```


Starting from the top, the `hdr` field contains metadata about the batch of packets. The `pkts` array stores opaque pointers to packet memory locations (e.g., CPU or GPU) across multiple segments, and `pkt_lens` stores the lengths of these packets. `pkt_extra_info` contains additional metadata about each packet, and `event` is a CUDA event used for synchronization.

As mentioned above, the `pkts` and `pkt_lens` fields are opaque and should not be accessed directly. Instead, refer to the helper functions in the next section for interacting with these fields to ensure compatibility with future versions.


#### Example API Usage

For an entire list of API functions, please see the `adv_network_common.h` header file.

##### Receive

The section below describes a workflowusing GPUDirect to receive packets using header-data split. The job of the user's operator(s)
is to process and free the buffers as quickly as possible. This might be copying to interim buffers or freeing before the entire
pipeline is done processing. This allows the networking piece to use relatively few buffers while still achieving very high rates.

The first step in receiving from the advanced network operator is to tie your operator's input port to the output port of the RX
network operator's `burst_out` port.

```
auto adv_net_rx    = make_operator<ops::AdvNetworkOpRx>("adv_network_rx", from_config("adv_network_common"), from_config("adv_network_rx"), make_condition<BooleanCondition>("is_alive", true));
auto my_receiver   = make_operator<ops::MyReceiver>("my_receiver", from_config("my_receiver"));
add_flow(adv_net_rx, my_receiver, {{"burst_out", "burst_in"}});
```

Once the ports are connected, inside the `compute()` function of your operator you will receive a `AdvNetBurstParams` structure
when a batch is complete:

```
auto burst = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in").value();
```

The packets arrive in scattered packet buffers. Depending on the application, you may need to iterate through the packets to
aggregate them into a single buffer. Alternatively the operator handling the packet data can operate on a list of packet
pointers rather than a contiguous buffer. Below is an example of aggregating separate GPU packet buffers into a single GPU
buffer:

```
  for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
    h_dev_ptrs_[aggr_pkts_recv_ + p]   = adv_net_get_cpu_pkt_ptr(burst, p);
    ttl_bytes_in_cur_batch_           += adv_net_get_gpu_packet_len(burst, p) + sizeof(UDPPkt);
  }

  simple_packet_reorder(buffer, h_dev_ptrs, packet_len, burst->hdr.num_pkts);
```

For this example we are tossing the header portion (CPU), so we don't need to examine the packets. Since we launched a reorder
kernel to aggregate the packets in GPU memory, we are also done with the GPU pointers. All buffers may be freed to the
advanced network operator at this point:

```
adv_net_free_all_burst_pkts_and_burst(burst_bufs_[b]);
```

##### Transmit

Transmitting packets works similar to the receive side, except the user is tasked with filling out the packets as much as it
needs to. As mentioned above, helper functions are available to fill in most boilerplate header information if that doesn't
change often.

Similar to the receive, the transmit operator needs to connect to `burst_in` on the advanced network operator transmitter:

```
auto my_transmitter  = make_operator<ops::MyTransmitter>("my_transmitter", from_config("my_transmitter"), make_condition<BooleanCondition>("is_alive", true));  
auto adv_net_tx       = make_operator<ops::AdvNetworkOpTx>("adv_network_tx", from_config("adv_network_common"), from_config("adv_network_tx"));
add_flow(my_transmitter, adv_net_tx, {{"burst_out", "burst_in"}});
```

Before sending packets, the user's transmit operator must request a buffer from the advanced network operator pool:

```
auto msg = std::make_shared<AdvNetBurstParams>();
msg->hdr.num_pkts = num_pkts;
if ((ret = adv_net_get_tx_pkt_burst(msg.get())) != AdvNetStatus::SUCCESS) {
  HOLOSCAN_LOG_ERROR("Error returned from adv_net_get_tx_pkt_burst: {}", static_cast<int>(ret));
  return;
}
```

The code above creates a shared `AdvNetBurstParams` that will be passed to the advanced network operator, and uses
`adv_net_get_tx_pkt_burst` to populate the burst buffers with valid packet buffers. On success, the buffers inside the
burst structure will be allocate and are ready to be filled in. Each packet must be filled in by the user. In this
example we loop through each packet and populate a buffer:

```
for (int num_pkt = 0; num_pkt < msg->hdr.num_pkts; num_pkt++) {
  void *payload_src = data_buf + num_pkt * nom_pkt_size;
  if (adv_net_set_udp_payload(msg->cpu_pkts[num_pkt], payload_src, nom_pkt_size) != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to create packet {}", num_pkt);
  }
}
```

The code iterates over `msg->hdr.num_pkts` (defined by the user) and passes a pointer to the payload and the packet
size to `adv_net_set_udp_payload`. In this example our configuration is using `fill_mode` "udp" on the transmitter, so
`adv_net_set_udp_payload` will populate the Ethernet, IP, and UDP headers. The payload pointer passed by the user
is also copied into the buffer. Alternatively a user could use the packet buffers directly as output from a previous stage
to avoid this extra copy.

With the `AdvNetBurstParams` populated, the burst can be sent off to the advanced network operator for transmission:

```
op_output.emit(msg, "burst_out");
```
