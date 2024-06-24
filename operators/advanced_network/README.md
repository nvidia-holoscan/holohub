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
- A DPDK-compatible network card. For GPUDirect only NVIDIA NICs are supported
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
./dev_container build --docker_file operators/advanced_network/Dockerfile --img holohub-doca:doca-27-ubuntu2204 --no-cache

# Launch DOCA container
./operators/advanced_network/run_doca.sh

# To build operator + app from main dir
./run build adv_networking_bench

# Run app
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_doca_tx_rx.yaml
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
only available at the boot command since they must be provided before the kernel has started. On the Clara AGX and Orin IGX
editing the boot command can be done with the following configuration:

```
sudo vim /boot/extlinux/extlinux.conf
# Find the line starting with APPEND and add the following

# For Orin IGX:
isolcpus=6-11 nohz_full=6-11 irqaffinity=0-5 rcu_nocbs=6-11 rcu_nocb_poll tsc=reliable audit=0 nosoftlockup default_hugepagesz=1G hugepagesz=1G hugepages=2

# For Clara AGX:
isolcpus=4-7 nohz_full=4=7 irqaffinity=0-3 rcu_nocbs=4-7 rcu_nocb_poll tsc=reliable audit=0 nosoftlockup default_hugepagesz=1G hugepagesz=1G hugepages=2
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
bound to a non-isolated core
  - type: `integer`

##### Receive Configuration

- **`if_name`**: Name of the interface or PCIe BDF to use
  - type: `string`
- **`queues`**: Array of queues
  - type: `array`
- **`name`**: Name of queue
  - type: `string`
- **`gpu_direct`**: GPUDirect is enabled on the queue
  - type: `boolean`
- **`batch_size`**: Number of packets in a batch that is passed between the advanced network operator and the user's operator. A
larger number increases throughput and latency by requiring fewer messages between operators, but takes longer to populate a single
buffer. A smaller number reduces latency and bandwidth by passing more messages.
- **`num_concurrent_batches`**: Number of batches that can be outstanding (not freed) at any given time. This value directly affects
the amount of memory needed for receiving packets. A value too small and packets will be dropped, while a value too large will
unnecessarily use excess CPU and/or GPU memory.
  - type: `integer`
- **`max_packet_size`**: Largest packet size expected
  - type: `integer`
- **`split_boundary`**: Split point in bytes where any byte before this value is sent to CPU, and anything after to GPU
  - type: `integer`
- **`gpu_device`**: GPU device number if using GPUDirect
  - type: `integer`
- **`cpu_cores`**: List of CPU cores from the isolated set used by the operator for receiving
  - type: `string`
- **`flows`**: Array of flows
  - type: `array`
- **`name`**: Name of queue
  - type: `string`
- **`action`**: Action section of flow
  - type: `sequence`
- **`type`**: Type of action. Only "queue" is supported currently.
  - type: `string`
- **`id`**: ID of queue to steer to
  - type: `integer`
- **`match`**: Match section of flow
  - type: `sequence`
- **`udp_src`**: UDP source port
  - type: `integer`
- **`udp_dst`**: UDP destination port
  - type: `integer`

##### Transmit Configuration

- **`if_name`**: Name of the interface or PCIe BDF to use
  - type: `string`
- **`accurate_send`**: Boolean flag to turn on accurate TX scheduling
  - type: `boolean`
- **`queues`**: Array of queues
  - type: `array`
- **`name`**: Name of queue
  - type: `string`
- **`id`**: ID of queue to steer to
  - type: `integer`
- **`gpu_direct`**: GPUDirect is enabled on the queue
  - type: `boolean`
- **`batch_size`**: Number of packets in a batch that is passed between the advanced network operator and the user's operator. A
larger number increases throughput and latency by requiring fewer messages between operators, but takes longer to populate a single
buffer. A smaller number reduces latency and bandwidth by passing more messages.
  - type: `integer`
- **`max_payload_size`**: Largest payload size expected
  - type: `integer`
- **`layer_fill`**: Layer(s) that the advanced network operator should populate in the packet. Anything higher than the layer
specified must be populated by the user. For example, if `ethernet` is specified, the user is responsible for populating values of
any item above that layer (IP, UDP, etc...). Valid values are `raw`, `ethernet`, `ip`, and `udp`
  - type: `string`
- **`eth_dst_addr`**: Destination ethernet MAC address. Only used for `ethernet` layer_fill mode or above
  - type: `string`
- **`ip_src_addr`**: Source IP address to send packets from. Only used for `ip` layer_fill and above
  - type: `string`
- **`ip_dst_addr`**: Destination IP address to send packets to. Only used for `ip` layer_fill and above
  - type: `string`
- **`udp_dst_port`**: UDP destination port. Only used for `udp` layer_fill and above
  - type: `integer`
- **`udp_src_port`**: UDP source port. Only used for `udp` layer_fill and above
  - type: `integer`
- **`cpu_cores`**: List of CPU cores for transmitting
  - type: `string`

  #### API Structures

  Both the transmit and receive operators use a common structure named `AdvNetBurstParams` to pass data to/from other operators.
  `AdvNetBurstParams` provides pointers to all packets on the CPU and GPU, and contains metadata needed by the operator to track
  allocations. Since the advanced network operator utilizes a generic interface that does not expose the underlying low-level network
  card library, interacting with the `AdvNetBurstParams` is mostly done with the helper functions described below. A user should
  never modify any members of `AdvNetBurstParams` directly as this may break in future versions. The `AdvNetBurstParams` is described
  below:

  ```
  struct AdvNetBurstParams {
    union {
        AdvNetBurstParamsHdr hdr;
        uint8_t buf[HS_NETWORK_HEADER_SIZE_BYTES];
    };

    void **cpu_pkts;
    void **gpu_pkts;
};
```

Starting from the top, the `hdr` field contains metadata about the batch of packets. `buf` is a placeholder for future expansion
of fields. `cpu_pkts` contains pointers to CPU packets, while `gpu_pkts` contains pointers to the GPU packets. As mentioned above,
the `cpu_pkts` and `gpu_pkts` are opaque pointers and should not be access directly. See the next section for information on interacting
with these fields.

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
