# High Performance Networking with Holoscan

This tutorial demonstrates how to use the Advanced Network library (referred to as `advanced_network` in HoloHub) for low latency and high throughput communication through NVIDIA SmartNICs. With a properly tuned system, the Advanced Network library can achieve hundreds of Gbps with latencies in the low microseconds.

!!! note

    This solution is designed for users who want to create a Holoscan application that will interface with an external system or sensor over Ethernet.

    - For high performance communication with systems also running Holoscan, refer to the [Holoscan distributed application documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_distributed_app.html) instead.
    - For JESD-compliant sensor without Ethernet support, consider the [Holoscan Sensor Bridge](https://docs.nvidia.com/holoscan/sensor-bridge/latest/introduction.html) for an FPGA-based interface to Holoscan.

## Prerequisites

Achieving High Performance Networking with Holoscan requires a system with an [**NVIDIA SmartNIC**](https://www.nvidia.com/en-us/networking/ethernet-adapters/) and a [**discrete GPU**](https://www.nvidia.com/en-us/design-visualization/desktop-graphics/). That is the case of [NVIDIA Data Center](https://www.nvidia.com/en-us/data-center/) systems, or edge systems like the [NVIDIA IGX](https://www.nvidia.com/en-us/edge-computing/products/igx/) platform and the [NVIDIA Project DIGITS](https://www.nvidia.com/en-us/project-digits/). `x86_64` systems equipped with these components are also supported, though the performance will vary greatly depending on the PCIe topology of the system (more on this [below](#31-ensure-ideal-pcie-topology)).

In this tutorial, we will be developing on an **NVIDIA IGX Orin platform** with [IGX SW 1.1](https://docs.nvidia.com/igx-orin/user-guide/latest/base-os.html) and an [NVIDIA RTX 6000 ADA GPU](https://www.nvidia.com/en-us/design-visualization/rtx-6000/), which is the configuration that is currently actively tested. The concepts should be applicable to other systems based on Ubuntu 22.04 as well. It should also work on other Linux distributions with a glibc version of 2.35 or higher by containerizing the dependencies and applications on top of an Ubuntu 22.04 image, but this is not actively tested at this time.

!!! Warning "Secure boot conflict"

    If you have secure boot enabled on your system, you might need to disable it as a prerequisite to run some of the configurations below ([switching the NIC link layers to Ethernet](#22-switch-your-nic-link-layers-to-ethernet), [updating the MRRS of your NIC ports](#33-maximize-the-nics-max-read-request-size-mrrs), [updating the BAR1 size of your GPU](#38-maximize-gpu-bar1-size)). Secure boot can be re-enabled after the configurations are completed.

## Background

Achieving high performance networking is a complex problem that involves many system components and configurations which we will cover in this tutorial. Two of the core concepts to achieve this are named Kernel Bypass, and GPUDirect.

### Kernel Bypass

In this context, Kernel Bypass refers to bypassing the operating system's kernel to directly communicate with the network interface (NIC), greatly reducing the latency and overhead of the Linux network stack. There are multiple technologies that achieve this in different fashions. They're all Ethernet-based, but differ in their implementation and features. The goal of the Advanced Network library in Holoscan Networking is to provide a common higher-level interface to all these backends:

- **RDMA**: Remote Direct Memory Access, using the open-source [`rdma-core`](https://github.com/linux-rdma/rdma-core) library. It differs from the other Ethernet-based backends with its server/client model and RoCE (RDMA over Ethernet) protocol. Given the extra cost and complexity to setup on both ends, it offers a simpler user interface, orders packets on arrival, and is the only one to offer a high reliability mode.
- **DPDK**: the Data Plane Development Kit is an open-source project part of the Linux Foundation with a strong and long-lasting community support. Its RTE Flow capability is generally considered the most flexible solution to split packets ingress and egress data.
- **DOCA GPUNetIO**: This NVIDIA proprietary technology differs from the other backends by transmitting and receiving packets from the NIC using a GPU kernel instead of CPU code, which is highly beneficial for CPU-bound applications.
- **NVIDIA Rivermax**: NVIDIA's other proprietary kernel bypass technology. For a license fee, it should offer the lowest latency and lowest resource utilization for video streaming (RTP packets).

??? example "Work In Progress"

    The Holoscan Advanced Network library integration testing infrastructure is under active development. As such:

    - The **DPDK** backend is supported and distributed with the `holoscan-networking` package, and is the only backend actively tested at this time.
    - The **DOCA GPUNetIO** backend is supported and distributed with the `holoscan-networking` package, with testing infrastructure under development.
    - The **NVIDIA Rivermax** backend is supported for Rx only when building from source, but not yet distributed nor actively tested. Tx support is under development.
    - The **RDMA** backend is under active development and should be available soon.

Which backend is best for your use case will depend on multiple factors, such as packet size, batch size, data type, and more. The goal of the Advanced Network library is to abstract the interface to these backends, allowing developers to focus on the application logic and experiment with different configurations to identify the best technology for their use case.

### GPUDirect

`GPUDirect` allows the NIC to read and write data from/to a GPU without requiring to copy the data the system memory, decreasing CPU overheads and significantly reducing latency. An implementation of `GPUDirect` is supported by all the kernel bypass backends listed above.

!!! Warning

    `GPUDirect` is only supported on Workstation/Quadro/RTX GPUs and Data Center GPUs. It is not supported on GeForce cards.

??? info "How does that relate to peermem or dma-buf?"

    There are two interfaces to enable `GPUDirect`:

    - The [`nvidia-peermem`](https://docs.nvidia.com/cuda/gpudirect-rdma/) kernel module, distributed with the NVIDIA DKMS GPU drivers.
        - Supported on Ubuntu kernels 5.4+, deprecated starting with kernel 6.8.
        - Supported on NVIDIA optimized Linux kernels, including IGX OS and DGX OS.
        - Supported by all MOFED drivers (requires rebuilding nvidia-dkms drivers afterwards).
    - [`DMA Buf`](https://docs.kernel.org/driver-api/dma-buf.html), supported on Linux kernels 5.12+ with NVIDIA open-source drivers 515+ and CUDA toolkit 11.7+.


## 1. Installing Holoscan Networking

We'll start with installing the `holoscan-networking` package, as it provides some utilities to help tune the system, and requires some dependencies which will help us with the system setup.

First, add the [DOCA apt repository](https://developer.nvidia.com/doca-downloads?deployment_platform=Host-Server&deployment_package=DOCA-Host&target_os=Linux) which holds some of its dependencies:

=== "IGX OS 1.1"

    ```bash
    export DOCA_URL="https://linux.mellanox.com/public/repo/doca/2.10.0/ubuntu22.04/arm64-sbsa/"
    wget -qO- https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub > /dev/null
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./"  | sudo tee /etc/apt/sources.list.d/doca.list > /dev/null

    sudo apt update
    ```

=== "SBSA (Ubuntu 22.04)"

    ```bash
    export DOCA_URL="https://linux.mellanox.com/public/repo/doca/2.10.0/ubuntu22.04/arm64-sbsa/"
    wget -qO- https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub > /dev/null
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./"  | sudo tee /etc/apt/sources.list.d/doca.list > /dev/null

    # Also need the CUDA repository for holoscan: https://developer.nvidia.com/cuda-downloads?target_os=Linux
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb

    sudo apt update
    ```

=== "x86_64 (Ubuntu 22.04)"

    ```bash
    export DOCA_URL="https://linux.mellanox.com/public/repo/doca/2.10.0/ubuntu22.04/x86_64/"
    wget -qO- https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub > /dev/null
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./"  | sudo tee /etc/apt/sources.list.d/doca.list > /dev/null

    # Also need the CUDA repository for holoscan: https://developer.nvidia.com/cuda-downloads?target_os=Linux
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb

    sudo apt update
    ```

You can then install `holoscan-networking`:

=== "Debian installation"

    ```bash
    sudo apt install -y holoscan-networking
    ```

=== "From source"

    You can build the Holoscan Networking libraries and sample applications from source on HoloHub:

    ```bash
    git clone git@github.com:nvidia-holoscan/holohub.git
    cd holohub
    ./holohub install holoscan-networking   # Installed in ./install
    ```

    If you'd like to generate the debian package from source and install it to ensure all dependencies are then present on your system, you can run:

    ```bash
    ./holohub install holoscan-networking
    sudo apt-get install ./holoscan-networking_*.deb        # Installed in /opt/nvidia/holoscan
    ```

    Refer to the [HoloHub README](https://github.com/nvidia-holoscan/holohub/blob/main/README.md) for more information.

## 2. Required System Setup

### 2.1 Check your NIC drivers

Ensure your NIC drivers are loaded:

```bash
lsmod | grep ib_core
```

??? abstract "See an example output"

    This would be an expected output, where `ib_core` is listed on the left.

    ```bash
    ib_core               442368  8 rdma_cm,ib_ipoib,iw_cm,ib_umad,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm
    mlx_compat             20480  11 rdma_cm,ib_ipoib,mlxdevm,iw_cm,ib_umad,ib_core,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm,mlx5_core
    ```

If this is empty, install the latest OFED drivers from DOCA (the DOCA APT repository should already be configured from the [Holoscan Networking installation above](#1-installing-holoscan-networking)), and reboot your system:

```bash
sudo apt update
sudo apt install doca-ofed
sudo reboot
```

!!! note

    If this is not empty, you can still install the newest OFED drivers from `doca-ofed` above. If you choose to keep your current drivers, install the following utilities for convenience later on. They include tools like `ibstat`, `ibv_devinfo`, `ibdev2netdev`, `mlxconfig`:

    ```bash
    sudo apt update
    sudo apt install infiniband-diags ibverbs-utils mlnx-ofed-kernel-utils mft
    ```

    Also upgrade the user space libraries to make sure your tools have all the symbols they need:

    ```bash
    sudo apt install libibverbs1 librdmacm1 rdma-core
    ```

Running `ibstat` or `ibv_devinfo` will confirm your NIC interfaces are recognized by your drivers.


### 2.2 Switch your NIC Link Layers to Ethernet

NVIDIA SmartNICs can function in two separate modes (called link layer):

- Ethernet (ETH)
- Infiniband (IB)

To identify the current mode, run `ibstat` or `ibv_devinfo` and look for the `Link Layer` value.

```bash
ibv_devinfo
```

??? failure "Couldn't load driver 'libmlx5-rdmav34.so'"

    If you see an error like this, you might have different versions for your OFED tools and libraries. Attempt after upgrading your user space libraries to match the version of your OFED tools like so:

    ```bash
    sudo apt update
    sudo apt install libibverbs1 librdmacm1 rdma-core
    ```

??? abstract "See an example output"

    In the example below, the `mlx5_0` interface is in Ethernet mode, while the `mlx5_1` interface is in Infiniband mode. Do not pay attention to the `transport` value which is always `InfiniBand`.

    ```sh hl_lines="18 37"
    hca_id: mlx5_0
            transport:                      InfiniBand (0)
            fw_ver:                         28.38.1002
            node_guid:                      48b0:2d03:00f4:07fb
            sys_image_guid:                 48b0:2d03:00f4:07fb
            vendor_id:                      0x02c9
            vendor_part_id:                 4129
            hw_ver:                         0x0
            board_id:                       NVD0000000033
            phys_port_cnt:                  1
                    port:   1
                            state:                  PORT_ACTIVE (4)
                            max_mtu:                4096 (5)
                            active_mtu:             4096 (5)
                            sm_lid:                 0
                            port_lid:               0
                            port_lmc:               0x00
                            link_layer:             Ethernet

    hca_id: mlx5_1
            transport:                      InfiniBand (0)
            fw_ver:                         28.38.1002
            node_guid:                      48b0:2d03:00f4:07fc
            sys_image_guid:                 48b0:2d03:00f4:07fb
            vendor_id:                      0x02c9
            vendor_part_id:                 4129
            hw_ver:                         0x0
            board_id:                       NVD0000000033
            phys_port_cnt:                  1
                    port:   1
                            state:                  PORT_ACTIVE (4)
                            max_mtu:                4096 (5)
                            active_mtu:             4096 (5)
                            sm_lid:                 0
                            port_lid:               0
                            port_lmc:               0x00
                            link_layer:             InfiniBand
    ```

**For Holoscan Networking, we want the NIC to use the ETH link layer.** To switch the link layer mode, there are two possible options:

1. On IGX Orin developer kits, you can switch that setting through the BIOS: [see IGX Orin documentation](https://docs.nvidia.com/igx-orin/user-guide/latest/switch-network-link.html).
2. On any system with a NVIDIA NIC (including the IGX Orin developer kits), you can run the commands below from a terminal:

    1. Identify the PCI address of your NVIDIA NIC

        === "ibdev2netdev"

            ```bash
            nic_pci=$(sudo ibdev2netdev -v | awk '{print $1}' | head -n1)
            ```

        === "lspci"

            ```bash
            # `0200` is the PCI-SIG class code for Ethernet controllers
            # `0207` is the PCI-SIG class code for Infiniband controllers
            # `15b3` is the Vendor ID for Mellanox
            nic_pci=$(lspci -n | awk '($2 == "0200:" || $2 == "0207:") && $3 ~ /^15b3:/ {print $1; exit}')
            ```

    2. Set both link layers to Ethernet. `LINK_TYPE_P1` and `LINK_TYPE_P2` are for `mlx5_0` and `mlx5_1` respectively. You can choose to only set one of them. `ETH` or `2` is Ethernet mode, and `IB` or `1` is for InfiniBand.

        ```bash
        sudo mlxconfig -d $nic_pci set LINK_TYPE_P1=ETH LINK_TYPE_P2=ETH
        ```

        Apply with `y`.

        ??? abstract "See an example output"

            ```sh
            Device #1:
            ----------

            Device type:    ConnectX7
            Name:           P3740-B0-QSFP_Ax
            Description:    NVIDIA Prometheus P3740 ConnectX-7 VPI PCIe Switch Motherboard; 400Gb/s; dual-port QSFP; PCIe switch5.0 X8 SLOT0 ;X16 SLOT2; secure boot;
            Device:         0005:03:00.0

            Configurations:                                      Next Boot       New
                    LINK_TYPE_P1                                ETH(2)          ETH(2)
                    LINK_TYPE_P2                                IB(1)           ETH(2)

            Apply new Configuration? (y/n) [n] :
            y

            Applying... Done!
            -I- Please reboot machine to load new configurations.
            ```

            - `Next Boot` is the current value that was expected to be used at the next reboot.
            - `New` is the value you're about to set to override `Next Boot`.

        ??? failure "ERROR: write counter to semaphore: Operation not permitted"

            Disable secure boot on your system ahead of changing the link type of your NIC ports. It can be re-enabled afterwards.

    3. Reboot your system.

        ```bash
        sudo reboot
        ```

### 2.3 Configure the IP addresses of the NIC ports

First, we want to identify the logical names of your NIC interfaces. Connecting an SFP cable in just one of the ports of the NIC will help you identify which port is which. Run the following command once the cable is in place:

```bash
ibdev2netdev
```

??? abstract "See an example output"

    In the example below, only `mlx5_1` has a cable connected (`Up`), and its logical ethernet name is `eth1`:

    ```bash
    $ ibdev2netdev
    mlx5_0 port 1 ==> eth0 (Down)
    mlx5_1 port 1 ==> eth1 (Up)
    ```

??? failure "ibdev2netdev does not show the NIC"

    If you have a cable connected but it does not show Up/Down in the output of `ibdev2netdev`, you can try to parse the output of `dmesg` instead. The example below shows that `0005:03:00.1` is plugged, and that it is associated with `eth1`:

    ```sh
    $ sudo dmesg | grep -w mlx5_core
    ...
    [   11.512808] mlx5_core 0005:03:00.0 eth0: Link down
    [   11.640670] mlx5_core 0005:03:00.1 eth1: Link down
    ...
    [ 3712.267103] mlx5_core 0005:03:00.1: Port module event: module 1, Cable plugged
    ```

The next step is to set a static IP on the interface you'd like to use so you can refer to it in your Holoscan applications. First, check if you already have any addresses configured using the ethernet interface names identified above (in our case, `eth0` and `eth1`):

```bash
ip -f inet addr show eth0
ip -f inet addr show eth1
```

If nothing appears, or you'd like to change the address, you can set an IP address through the Network Manager user interface, CLI (`nmcli`), or other IP configuration tools. In the example below, we configure the `eth0` interface with an address of `1.1.1.1/24`, and the `eth1` interface with an address of `2.2.2.2/24`.

=== "One-time"

    ```bash
    sudo ip addr add 1.1.1.1/24 dev eth0
    sudo ip addr add 2.2.2.2/24 dev eth1
    ```

=== "Persistent"

    Set these variables to your desired values:

    ```bash
    if_name=eth0
    if_static_ip=1.1.1.1/24
    ```

    === "NetworkManager"

        Update the IP with `nmcli`:

        ```bash
        sudo nmcli connection modify $if_name ipv4.addresses $if_static_ip
        sudo nmcli connection up $if_name
        ```

    === "systemd-networkd"

        Create a network config file with the static IP:

        ```bash
        cat << EOF | sudo tee /etc/systemd/network/20-$if_name.network
        [Match]
        MACAddress=$(cat /sys/class/net/$if_name/address)

        [Network]
        Address=$if_static_ip
        EOF
        ```

        Apply now:

        ```bash
        sudo systemctl restart systemd-networkd
        ```

!!! note

    If you are connecting the NIC to another NIC with an [interconnect](https://www.nvidia.com/en-us/networking/interconnect/), do the same on the other system with an IP address on the same network segment.
    For example, to communicate with `1.1.1.1/24` above (`/24` -> `255.255.255.0` submask), setup your other system with an IP between `1.1.1.2` and `1.1.1.254`, and the same `/24` submask.

### 2.4 Enable GPUDirect

Assuming you already have [NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installation-network) installed, check if the `nvidia_peermem` kernel module is loaded:

=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check topo
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check topo

        ```

    ??? abstract "See an example output"

        ```log
        2025-03-12 14:15:07 - INFO - GPU 0: NVIDIA RTX A6000 has GPUDirect support.
        2025-03-12 14:15:27 - INFO - nvidia-peermem module is loaded.
        ```

```bash
lsmod | grep nvidia_peermem
```

If it's not loaded, run the following command, then check again:

=== "One-time"

    ```bash
    sudo modprobe nvidia_peermem
    ```

=== "Persistent"

    ```bash
    sudo echo "nvidia-peermem" >> /etc/modules
    sudo systemctl restart systemd-modules-load.service
    ```

??? failure "Error loading the `nvidia-peermem` kernel module"

    If you run into an error loading the `nvidia-peermem` kernel module, follow these steps:

    1. Install the `doca-ofed` package to get the latest drivers for your NIC as [documented above](#21-check-your-nic-drivers).
    2. Restart your system.
    3. Rebuild your NVIDIA drivers with DKMS like so:

    ```bash
    peermem_ko=$(find /lib/modules/$(uname -r) -name "*peermem*.ko")
    nv_dkms=$(dpkg -S "$peermem_ko" | cut -d: -f1)
    sudo dpkg-reconfigure $nv_dkms
    sudo modprobe nvidia_peermem
    ```

??? info "Why peermem and not dma buf?"

    `peermem` is currently the only GPUDirect interface supported by all our [networking backends](#kernel-bypass). This section will therefore provide instructions for `peermem` and not `dma buf`.

## 3. Optimal system configurations

!!! warning "Advanced"

    The section below is for advanced users looking to extract more performance out of their system. You can choose to skip this section and return to it later if performance if your application is not satisfactory.

While the configurations above are the minimum requirements to get a NIC and a NVIDIA GPU to communicate while bypassing the OS kernel stack, performance can be further improved in most scenarios by tuning the system as described below.

Before diving in each of the setups below, we provide a utility script as part of the `holoscan-networking` package which provides an overview of the configurations that potentially need to be tuned on your system.

??? example "Work In Progress"

    This utility script is under active development and will be updated in future releases with additional checks, more actionable recommendations, and automated tuning.

=== "Debian installation"

    ```bash
    sudo /opt/nvidia/holoscan/bin/tune_system.py --check all
    ```

=== "From source"

    ```bash
    cd holohub
    sudo ./operators/advanced_network/python/tune_system.py --check all
    ```

??? abstract "See an example output"

    Our tuned-up IGX system with A6000 can optimize most settings:

    ```log
    2025-03-12 14:16:06 - INFO - CPU 0: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 1: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 2: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 3: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 4: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 5: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 6: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 7: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 8: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 9: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 10: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - CPU 11: Governor is correctly set to 'performance'.
    2025-03-12 14:16:06 - INFO - cx7_0/0005:03:00.0: MRRS is correctly set to 4096.
    2025-03-12 14:16:06 - INFO - cx7_1/0005:03:00.1: MRRS is correctly set to 4096.
    2025-03-12 14:16:06 - WARNING - cx7_0/0005:03:00.0: PCIe Max Payload Size is not set to 256 bytes. Found: 128 bytes.
    2025-03-12 14:16:06 - WARNING - cx7_1/0005:03:00.1: PCIe Max Payload Size is not set to 256 bytes. Found: 128 bytes.
    2025-03-12 14:16:06 - INFO - HugePages_Total: 3
    2025-03-12 14:16:06 - INFO - HugePage Size: 1024.00 MB
    2025-03-12 14:16:06 - INFO - Total Allocated HugePage Memory: 3072.00 MB
    2025-03-12 14:16:06 - INFO - Hugepages are sufficiently allocated with at least 500 MB.
    2025-03-12 14:16:06 - INFO - GPU 0: SM Clock is correctly set to 1920 MHz (within 500 of the 2100 MHz theoretical Max).
    2025-03-12 14:16:06 - INFO - GPU 0: Memory Clock is correctly set to 8000 MHz.
    2025-03-12 14:16:06 - INFO - GPU 00000005:09:00.0: BAR1 size is 8192 MiB.
    2025-03-12 14:16:06 - INFO - GPU GPU0 has at least one PIX/PXB connection to a NIC
    2025-03-12 14:16:06 - INFO - isolcpus found in kernel boot line
    2025-03-12 14:16:06 - INFO - rcu_nocbs found in kernel boot line
    2025-03-12 14:16:06 - INFO - irqaffinity found in kernel boot line
    2025-03-12 14:16:06 - INFO - Interface cx7_0 has an acceptable MTU of 9000 bytes.
    2025-03-12 14:16:06 - INFO - Interface cx7_1 has an acceptable MTU of 9000 bytes.
    2025-03-12 14:16:06 - INFO - GPU 0: NVIDIA RTX A6000 has GPUDirect support.
    2025-03-12 14:16:06 - INFO - nvidia-peermem module is loaded.
    ```

Based on the results, you can figure out which of the sections below are appropriate to update configurations on your system.

### 3.1 Ensure ideal PCIe topology

Kernel bypass and GPUDirect rely on PCIe to communicate between the GPU and the NIC at high speeds. As-such, the topology of the PCIe tree on a system is critical to ensure optimal performance.

Run the following command to check the GPUDirect communication matrix. **You are looking for a `PXB` or `PIX` connection between the GPU and the NIC interfaces to get the best performance.**

=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check topo
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check topo
        ```

    ??? abstract "See an example output"

        On IGX developer kits, the board's internal switch is designed to connect the GPU to the NIC interfaces with a `PXB` connection, offering great performance.

        ```log
        2025-03-06 12:07:45 - INFO - GPU GPU0 has at least one PIX/PXB connection to a NIC
        ```

=== "nvidia-smi"

    ```bash
    nvidia-smi topo -mp
    ```

    ??? abstract "See an example output"

        On IGX developer kits, the board's internal switch is designed to connect the GPU to the NIC interfaces with a `PXB` connection, offering great performance.
        ```
                GPU0    NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
        GPU0     X      PXB     PXB     0-11    0               N/A
        NIC0    PXB      X      PIX
        NIC1    PXB     PIX      X

        Legend:

        X    = Self
        SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
        NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
        PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
        PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
        PIX  = Connection traversing at most a single PCIe bridge

        NIC Legend:

        NIC0: mlx5_0
        NIC1: mlx5_1
        ```

If your connection is not optimal, you might be able to improve it by moving your NIC and/or GPU on a different PCIe port, so that they can share a branch and do not require going back to the Host Bridge (the CPU) to communicate. Refer to your system manufacturer for documentation, or run the following command to inspect the topology of your system:

```bash
lspci -tv
```

??? abstract "See an example output"

    Here is the PCIe tree of an IGX system. Note how the ConnectX-7 and RTX A6000 are connected to the same branch.
    ``` hl_lines="2 3 5"
    -+-[0007:00]---00.0-[01-ff]----00.0  Marvell Technology Group Ltd. 88SE9235 PCIe 2.0 x2 4-port SATA 6 Gb/s Controller
    +-[0005:00]---00.0-[01-ff]----00.0-[02-09]--+-00.0-[03]--+-00.0  Mellanox Technologies MT2910 Family [ConnectX-7]
    |                                           |            \-00.1  Mellanox Technologies MT2910 Family [ConnectX-7]
    |                                           +-01.0-[04-06]----00.0-[05-06]----08.0-[06]--
    |                                           \-02.0-[07-09]----00.0-[08-09]----00.0-[09]--+-00.0  NVIDIA Corporation GA102GL [RTX A6000]
    |                                                                                        \-00.1  NVIDIA Corporation GA102 High Definition Audio Controller
    +-[0004:00]---00.0-[01-ff]----00.0  Sandisk Corp WD PC SN810 / Black SN850 NVMe SSD
    +-[0001:00]---00.0-[01-ff]----00.0-[02-fc]--+-01.0-[03-34]----00.0  Realtek Semiconductor Co., Ltd. RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller
    |                                           +-02.0-[35-66]----00.0  Realtek Semiconductor Co., Ltd. RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller
    |                                           +-03.0-[67-98]----00.0  Device 1c00:3450
    |                                           +-04.0-[99-ca]----00.0-[9a]--+-00.0  ASPEED Technology, Inc. ASPEED Graphics Family
    |                                           |                            \-02.0  ASPEED Technology, Inc. Device 2603
    |                                           \-05.0-[cb-fc]----00.0  Realtek Semiconductor Co., Ltd. RTL8822CE 802.11ac PCIe Wireless Network Adapter
    \-[0000:00]-
    ```

!!! warning "x86_64 compatibility"

    Most x86_64 systems are not designed for this topology as they lack a discrete PCIe switch. In that case, the best connection they can achieve is `NODE`.


### 3.2 Check the NIC's PCIe configuration

!!! quote "[Understanding PCIe Configuration for Maximum Performance - May 27, 2022](https://enterprise-support.nvidia.com/s/article/understanding-pcie-configuration-for-maximum-performance)"

    PCIe is used in any system for communication between different modules [including the NIC and the GPU]. This means that in order to process network traffic, the different devices communicating via the PCIe should be well configured. When connecting the network adapter to the PCIe, it auto-negotiates for the maximum capabilities supported between the network adapter and the CPU.

The instructions below are meant to understand if your system is able to extract the maximum capabilities of your NIC, but they're not configurable. The two values that we are looking at here are the Max Payload Size (MPS - the maximum size of a PCIe packet) and the Speed (or PCIe generation).

##### Max Payload Size (MPS)

=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check mps
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check mps
        ```

    ??? abstract "See an example output"

        The PCIe configuration on the IGX Orin developer kit is not able to leverage the max payload size of the NIC:

        ```log
        2025-03-10 16:15:54 - WARNING - cx7_0/0005:03:00.0: PCIe Max Payload Size is not set to 256 bytes. Found: 128 bytes.
        2025-03-10 16:15:54 - WARNING - cx7_1/0005:03:00.1: PCIe Max Payload Size is not set to 256 bytes. Found: 128 bytes.
        ```

=== "manual"

    Identify the PCIe address of your NVIDIA NIC:

    === "ibdev2netdev"

        ```bash
        nic_pci=$(sudo ibdev2netdev -v | awk '{print $1}' | head -n1)
        ```

    === "lspci"

        ```bash
        # `0200` is the PCI-SIG class code for NICs
        # `15b3` is the Vendor ID for Mellanox
        nic_pci=$(lspci -n | awk '$2 == "0200:" && $3 ~ /^15b3:/ {print $1}' | head -n1)
        ```

    Check current and max MPS:

    ```bash
    sudo lspci -vv -s $nic_pci | awk '/DevCap/{s=1} /DevCtl/{s=0} /MaxPayload /{match($0, /MaxPayload [0-9]+/, m); if(s){print "Max " m[0]} else{print "Current " m[0]}}'
    ```

    ??? abstract "See an example output"

        The PCIe configuration on the IGX Orin developer kit is not able to leverage the max payload size of the NIC:

        ```log
        Max MaxPayload 512
        Current MaxPayload 128
        ```

    !!! note

        While your NIC might be capable of more, 256 bytes is generally the largest supported by any switch/CPU at this time.


##### PCIe Speed/Generation

Identify the PCIe address of your NVIDIA NIC:

=== "ibdev2netdev"

    ```bash
    nic_pci=$(sudo ibdev2netdev -v | awk '{print $1}' | head -n1)
    ```

=== "lspci"

    ```bash
    # `0200` is the PCI-SIG class code for NICs
    # `15b3` is the Vendor ID for Mellanox
    nic_pci=$(lspci -n | awk '$2 == "0200:" && $3 ~ /^15b3:/ {print $1}' | head -n1)
    ```

Check current and max Speeds:

```bash
sudo lspci -vv -s $nic_pci | awk '/LnkCap/{s=1} /LnkSta/{s=0} /Speed /{match($0, /Speed [0-9]+GT\/s/, m); if(s){print "Max " m[0]} else{print "Current " m[0]}}'
```

??? abstract "See an example output"

    On IGX, the switch is able to maximize the NIC speed, both being PCIe 5.0:

    ```log
    Max Speed 32GT/s
    Current Speed 32GT/s
    ```

### 3.3 Maximize the NIC's Max Read Request Size (MRRS)

!!! quote "[Understanding PCIe Configuration for Maximum Performance - May 27, 2022](https://enterprise-support.nvidia.com/s/article/understanding-pcie-configuration-for-maximum-performance)"

    PCIe Max Read Request determines the maximal PCIe read request allowed. A PCIe device usually keeps track of the number of pending read requests due to having to prepare buffers for an incoming response. The size of the PCIe max read request may affect the number of pending requests (when using data fetch larger than the PCIe MTU).

Unlike the PCIe properties queried in the previous section, the MRRS is configurable. **We recommend maxing it to 4096 bytes**. Run the following to check your current settings:

=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check mrrs
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check mrrs
        ```


=== "manual"

    Identify the PCIe address of your NVIDIA NIC:

    === "ibdev2netdev"

        ```bash
        nic_pci=$(sudo ibdev2netdev -v | awk '{print $1}' | head -n1)
        ```

    === "lspci"

        ```bash
        # `0200` is the PCI-SIG class code for NICs
        # `15b3` is the Vendor ID for Mellanox
        nic_pci=$(lspci -n | awk '$2 == "0200:" && $3 ~ /^15b3:/ {print $1}' | head -n1)
        ```

    Check current MRRS:

    ```bash
    sudo lspci -vv -s $nic_pci | grep DevCtl: -A2 | grep -oE "MaxReadReq [0-9]+"
    ```

Update MRRS:

=== "Debian installation"

    ```bash
    sudo /opt/nvidia/holoscan/bin/tune_system.py --set mrrs
    ```

=== "From source"

    ```bash
    cd holohub
    sudo ./operators/advanced_network/python/tune_system.py --set mrrs
    ```

!!! note

    This value is reset on reboot and needs to be set every time the system boots

??? failure "ERROR: pcilib: sysfs_write: write failed: Operation not permitted"

    Disable secure boot on your system ahead of changing the MRRS of your NIC ports. It can be re-enabled afterwards.

### 3.4 Enable Huge pages

Huge pages are a memory management feature that allows the OS to allocate large blocks of memory (typically 2MB or 1GB) instead of the default 4KB pages. This reduces the number of page table entries and the amount of memory used for translation, improving cache performance and reducing TLB (Translation Lookaside Buffer) misses, which leads to lower latencies.

While it is naturally beneficial for CPU packets, it is also needed when routing data packets to the GPU in order to handle metadata (mbufs) on the CPU.

=== "hugeadm"

    We recommend installing the `libhugetlbfs-bin` package for the `hugeadm` utility:

    ```bash
    sudo apt update
    sudo apt install -y libhugetlbfs-bin
    ```

    Then, check your huge page pools:

    ```bash
    hugeadm --pool-list
    ```

    ??? abstract "See an example output"

        The example below shows that this system supports huge pages of 64K, 2M (default), 32M, and 1G, but that none of them are currently allocated.

        ```
              Size  Minimum  Current  Maximum  Default
             65536        0        0        0
           2097152        0        0        0        *
          33554432        0        0        0
        1073741824        0        0        0
        ```

    And your huge page mount points:

    ```bash
    hugeadm --list-all-mounts
    ```

    ??? abstract "See an example output"

        The default huge pages are mounted on `/dev/hugepages` with a page size of 2M.

        ```
        Mount Point          Options
        /dev/hugepages       rw,relatime,pagesize=2M
        ```


=== "vanilla"

    First, check your huge page pools:

    ```bash
    ls -1 /sys/kernel/mm/hugepages/
    grep Huge /proc/meminfo
    ```

    ??? abstract "See an example output"

        The example below shows that this system supports huge pages of 64K, 2M (default), 32M, and 1G, but that none of them are currently allocated.

        ```
        hugepages-1048576kB
        hugepages-2048kB
        hugepages-32768kB
        hugepages-64kB
        ```

        ```
        HugePages_Total:       0
        HugePages_Free:        0
        HugePages_Rsvd:        0
        HugePages_Surp:        0
        Hugepagesize:       2048 kB
        Hugetlb:               0 kB
        ```

    And your huge page mount points:

    ```bash
    mount | grep huge
    ```

    ??? abstract "See an example output"

        The default huge pages are mounted on `/dev/hugepages` with a page size of 2M.

        ```
        hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,pagesize=2M)
        ```

**As a rule of thumb, we recommend to start with 3 to 4 GB of total huge pages, with an individual page size of 500 MB to 1 GB** (per system availability).

There are two ways to allocate huge pages:

- in the kernel bootline (recommended to ensure contiguous memory allocation) or
- dynamically at runtime (risk of fragmentation for large page sizes)

The example below allocates 3 huge pages of 1GB each.

=== "Kernel bootline"

    Add the flags below to the `GRUB_CMDLINE_LINUX` variable in `/etc/default/grub`:

    ```bash
    default_hugepagesz=1G hugepagesz=1G hugepages=3
    ```

    ??? info "Show explanation"

        - `default_hugepagesz`: the default huge page size to use, making them available from the default mount point, `/dev/hugepages`.
        - `hugepagesz`: the size of the huge pages to allocate.
        - `hugepages`: the number of huge pages to allocate.

    Then rebuild your GRUB configuration and reboot:

    ```bash
    sudo update-grub
    sudo reboot
    ```

=== "Runtime"

    Allocate the 3x 1GB huge pages:

    === "hugeadm"

        ```bash
        sudo hugeadm --pool-pages-min 1073741824:3
        ```

    === "vanilla"

        ```bash
        echo 3 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
        ```

    Create a mount point to access the 1GB huge pages pool since that is not the default size on that system. We will name it `/mnt/huge` here.

    === "One-time"

        ```bash
        sudo mkdir -p /mnt/huge
        sudo mount -t hugetlbfs -o pagesize=1G none /mnt/huge
        ```

    === "Persistent"

        ```bash
        echo "nodev /mnt/huge hugetlbfs pagesize=1G 0 0" | sudo tee -a /etc/fstab
        sudo mount /mnt/huge
        ```

    !!! note

        If you work with containers, remember to mount this directory in your container as well with `-v /mnt/huge:/mnt/huge`.


Rerunning the initial commands should now list 3 hugepages of 1GB each. 1GB will be the default huge page size if updated in the kernel bootline only.

### 3.5 Isolate CPU cores

!!! note

    This optimization is less impactful when using the `gpunetio` backend since the GPU polls the NIC.

The CPU interacting with the NIC to route packets is sensitive to perturbations, especially with smaller packet/batch sizes requiring more frequent work. Isolating a CPU in Linux prevents unwanted user or kernel threads from running on it, reducing context switching and latency spikes from noisy neighbors.

We recommend isolating the CPU cores you will select to interact with the NIC (defined in the `advanced_network` configuration [described later](#51-understand-the-configuration-parameters) in this tutorial). This is done by setting additional flags on the kernel bootline.

You can first check if any of the recommended flags were already set on the last boot:

=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check cmdline
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check cmdline
        ```

=== "manual"

    ```bash
    cat /proc/cmdline | grep -e isolcpus -e irqaffinity -e nohz_full -e rcu_nocbs -e rcu_nocb_poll
    ```

Decide which cores to isolate based on your configuration. We recommend one core per queue as a rule of thumb. First, identify your core IDs:

```bash
cat /proc/cpuinfo | grep processor
```

??? abstract "See an example output"

    This system has 12 cores, numbered 0 to 11:
    ```bash
    processor       # 0
    processor       # 1
    processor       # 2
    processor       # 3
    processor       # 4
    processor       # 5
    processor       # 6
    processor       # 7
    processor       # 8
    processor       # 9
    processor       # 10
    processor       # 11
    ```

As an example, the line below will isolate cores 9, 10 and 11, leaving cores 0-8 free for other tasks and hardware interrupts:

```bash
isolcpus=9-11 irqaffinity=0-8 nohz_full=9-11 rcu_nocbs=9-11 rcu_nocb_poll
```

??? info "Show explanation"

    | Parameter | Description |
    | --------- | ----------- |
    | `isolcpus` | Isolates specific CPU cores from the Linux scheduler, preventing regular system tasks from running on them. This ensures dedicated cores are available exclusively for your networking tasks, reducing context switches and interruptions that can cause latency spikes. |
    | `irqaffinity` | Controls which CPU cores can handle hardware interrupts. By directing network interrupts away from your isolated cores, you prevent networking tasks from being interrupted by hardware events, maintaining consistent processing time. |
    | `nohz_full` | Disables regular kernel timer ticks on specified cores when they're running user space applications. This reduces overhead and prevents periodic interruptions, allowing your networking code to run with fewer disturbances. |
    | `rcu_nocbs` | Offloads Read-Copy-Update (RCU) callback processing from specified cores. RCU is a synchronization mechanism in the Linux kernel that can cause periodic processing bursts. Moving this work away from your networking cores helps maintain consistent performance. |
    | `rcu_nocb_poll` | Works with `rcu_nocbs` to improve how RCU callbacks are processed on non-callback CPUs. This can reduce latency spikes by changing how the kernel polls for RCU work. |

    Together, these parameters create an environment where specific CPU cores can focus exclusively on network packet processing with minimal interference from the operating system, resulting in lower and more consistent latency.

Add these flags to the `GRUB_CMDLINE_LINUX` variable in `/etc/default/grub`, then rebuild your GRUB configuration and reboot:

```bash
sudo update-grub
sudo reboot
```

Verify that the flags were properly set after boot by rerunning the check commands above.

### 3.6 Prevent CPU cores from going idle

When a core goes idle/to sleep, coming back online to poll the NIC can cause latency spikes and dropped packets. To prevent this, **we recommend setting the scaling governor to `performance` for these CPU cores**.

!!! note

    Cores from a single cluster will always share the same governor.

!!! bug

    We have witnessed instances where setting the governor to `performance` on only the isolated cores (dedicated to polling the NIC) does not lead to the performance gains expected. As such, we currently recommend setting the governor to `performance` for all cores which has shown to be reliably effective.

Check the current governor for each of your cores:


=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check cpu-freq
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check cpu-freq
        ```

    ??? abstract "See an example output"

        ```
        2025-03-06 12:20:27 - WARNING - CPU 0: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 1: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 2: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 3: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 4: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 5: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 6: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 7: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 8: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 9: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 10: Governor is set to 'powersave', not 'performance'.
        2025-03-06 12:20:27 - WARNING - CPU 11: Governor is set to 'powersave', not 'performance'.
        ```

=== "manual"

    ```bash
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    ```

    ??? abstract "See an example output"

        In this example, all cores were defaulted to `powersave` instead of the recommended `performance`.

        ```
        powersave
        powersave
        powersave
        powersave
        powersave
        powersave
        powersave
        powersave
        powersave
        powersave
        powersave
        powersave
        ```

Install `cpupower` to more conveniently set the governor:

```bash
sudo apt update
sudo apt install -y linux-tools-$(uname -r)
```

Set the governor to `performance` for all cores:

=== "One-time"

    ```bash
    sudo cpupower frequency-set -g performance
    ```

=== "Persistent"

    ```bash
    cat << EOF | sudo tee /etc/systemd/system/cpu-performance.service
    [Unit]
    Description=Set CPU governor to performance
    After=multi-user.target

    [Service]
    Type=oneshot
    ExecStart=/usr/bin/cpupower -c all frequency-set -g performance

    [Install]
    WantedBy=multi-user.target
    EOF
    sudo systemctl enable cpu-performance.service
    sudo systemctl start cpu-performance.service
    ```

Running the checks above should now list `performance` as the governor for all cores. You can also run `sudo cpupower -c all frequency-info` for more details.

### 3.7 Prevent the GPU from going idle

Similarly to the above, we want to maximize the GPU's clock speed and prevent it from going idle.

Run the following command to check your current clocks and whether they're locked (persistence mode):

```
nvidia-smi -q | grep -i "Persistence Mode"
nvidia-smi -q -d CLOCK
```

??? abstract "See an example output"

    ``` hl_lines="1 7 8 20 21"
        Persistence Mode: Enabled
    ...
    Attached GPUs                             : 1
    GPU 00000005:09:00.0
        Clocks
            Graphics                          : 420 MHz
            SM                                : 420 MHz
            Memory                            : 405 MHz
            Video                             : 1680 MHz
        Applications Clocks
            Graphics                          : 1800 MHz
            Memory                            : 8001 MHz
        Default Applications Clocks
            Graphics                          : 1800 MHz
            Memory                            : 8001 MHz
        Deferred Clocks
            Memory                            : N/A
        Max Clocks
            Graphics                          : 2100 MHz
            SM                                : 2100 MHz
            Memory                            : 8001 MHz
            Video                             : 1950 MHz
        ...
    ```


To lock the GPU's clocks to their max values:

=== "One-time"

    ```bash
    sudo nvidia-smi -pm 1
    sudo nvidia-smi -lgc=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
    sudo nvidia-smi -lmc=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits)
    ```

=== "Persistent"

    ```bash
    cat << EOF | sudo tee /etc/systemd/system/gpu-max-clocks.service
    [Unit]
    Description=Max GPU clocks
    After=multi-user.target

    [Service]
    Type=oneshot
    ExecStart=/usr/bin/nvidia-smi -pm 1
    ExecStart=/bin/bash -c '/usr/bin/nvidia-smi --lock-gpu-clocks=$(/usr/bin/nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)'
    ExecStart=/bin/bash -c '/usr/bin/nvidia-smi --lock-memory-clocks=$(/usr/bin/nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits)'
    RemainAfterExit=true

    [Install]
    WantedBy=multi-user.target
    EOF

    sudo systemctl enable gpu-max-clocks.service
    sudo systemctl start gpu-max-clocks.service
    ```

??? info "Show explanation"

    This queries the max clocks for the GPU SM (`clocks.max.sm`) and memory (`clocks.max.mem`) and sets them to the current clocks (`lock-gpu-clocks` and `lock-memory-clocks` respectively). `-pm 1` (or `--persistence-mode=1`) enables persistence mode to lock these values.

??? abstract "See an example output"

    ```
    GPU clocks set to "(gpuClkMin 2100, gpuClkMax 2100)" for GPU 00000005:09:00.0
    All done.
    Memory clocks set to "(memClkMin 8001, memClkMax 8001)" for GPU 00000005:09:00.0
    All done.
    ```

You can confirm that the clocks are set to the max values by running `nvidia-smi -q -d CLOCK` again.

!!! note

    Some max clocks might not be achievable in certain configurations, or due to boost clocks (SM) or rounding errors (Memory),  despite the lock commands indicating it worked. For example - on IGX - the max non-boot SM clock will be 1920 MHz, and the max memory clock will show 8000 MHz, which are satisfying compared to the initial mode.


### 3.8 Maximize GPU BAR1 size

The GPU BAR1 memory is the primary resource consumed by `GPUDirect`. It allows other PCIe devices (like the CPU and the NIC) to access the GPU's memory space. The larger the BAR1 size, the more memory the GPU can expose to these devices in a single PCIe transaction, reducing the number of transactions needed and improving performance.

**We recommend a BAR1 size of 1GB or above.** Check the current BAR1 size:

=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check bar1-size
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check bar1-size
        ```

    ??? abstract "See an example output"

        ```
        2025-03-06 12:22:53 - INFO - GPU 00000005:09:00.0: BAR1 size is 8192 MiB.
        ```

=== "manual"

    ```bash
    nvidia-smi -q | grep -A 3 BAR1
    ```

    ??? abstract "See an example output"

        For our RTX A6000, this shows a BAR1 size of 256 MiB:

        ```
            BAR1 Memory Usage
            Total                             : 256 MiB
            Used                              : 13 MiB
            Free                              : 243 MiB
        ```

!!! warning

    Resizing the BAR1 size requires:

    - A BIOS with resizable BAR support
    - A GPU with physical resizable BAR

    **If you attempt to go forward with the instructions below without meeting the above requirements, you might render your GPU unusable.**

##### BIOS Resizable BAR support

First, check if your system and BIOS support resizable BAR. Refer to your system's manufacturer documentation to access the BIOS. The Resizable BAR option is often categorized under `Advanced > PCIe` settings. Enable this feature if found.

!!! note

    The IGX Developer kit with IGX OS 1.1+ supports resizable BAR by default.

##### GPU Resizable BAR support

Next, you can check if your GPU has physical resizable BAR by running the following command:

```bash
sudo lspci -vv -s $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader) | grep BAR
```

??? abstract "See an example output"

    This RTX A6000 has a resizable BAR1, currently set to 256 MiB:

    ```
    Capabilities: [bb0 v1] Physical Resizable BAR
        BAR 0: current size: 16MB, supported: 16MB
        BAR 1: current size: 256MB, supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB 64GB
        BAR 3: current size: 32MB, supported: 32MB
    ```

If your GPU is listed [on this page](https://developer.nvidia.com/displaymodeselector), you can download the `Display Mode Selector` to resize the BAR1 to 8GB.

1. Press `Join Now`.
2. Once approved, download the `Display Mode Selector` archive.
3. Unzip the archive.
4. Access your system without a X-server running, either through SSH or a Virtual Console (`Alt+F1`).
5. Go down the right OS and architecture folder for your system (`linux/aarch64` or `linux/x64`).
6. Run the `displaymodeselector` command like so:

```bash
chmod +x displaymodeselector
sudo ./displaymodeselector --gpumode physical_display_enabled_8GB_bar1
```

Press `y` to confirm you'd like to continue, then `y` again to apply to all the eligible adapters.

??? abstract "See an example output"

    ```
    NVIDIA Display Mode Selector Utility (Version 1.67.0)
    Copyright (C) 2015-2021, NVIDIA Corporation. All Rights Reserved.

    WARNING: This operation updates the firmware on the board and could make
            the device unusable if your host system lacks the necessary support.

    Are you sure you want to continue?
    Press 'y' to confirm (any other key to abort):
    y
    Specified GPU Mode "physical_display_enabled_8GB_bar1"


    Update GPU Mode of all adapters to "physical_display_enabled_8GB_bar1"?
    Press 'y' to confirm or 'n' to choose adapters or any other key to abort:
    y

    Updating GPU Mode of all eligible adapters to "physical_display_enabled_8GB_bar1"

    Apply GPU Mode <6> corresponds to "physical_display_enabled_8GB_bar1"

    Reading EEPROM (this operation may take up to 30 seconds)

    [==================================================] 100 %
    Reading EEPROM (this operation may take up to 30 seconds)

    Successfully updated GPU mode to "physical_display_enabled_8GB_bar1" ( Mode 6 ).

    A reboot is required for the update to take effect.
    ```

??? failure "Error: unload the NVIDIA kernel driver first"

    If you see this error:

    ```bash
    ERROR: In order to avoid the irreparable damage to your graphics adapter it is necessary to unload the NVIDIA kernel driver first:

    rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia_peermem nvidia
    ```

    Try to unload the NVIDIA kernel driver listed in the error message above (list may vary):

    ```bash
    sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia_peermem nvidia
    ```

    If this fails because the drivers are in use, stop the X-server first before trying again:

    ```bash
    sudo systemctl isolate multi-user
    ```

??? failure "/dev/mem: Operation not permitted. Access to physical memory denied"

    Disable secure boot on your system ahead of changing your GPU's BAR1 size. It can be re-enabled afterwards.

Reboot your system, and check the BAR1 size again to confirm the change.

```
sudo reboot
```

### 3.9 Enable Jumbo Frames

Jumbo frames are Ethernet frames that carry a payload larger than the standard 1500 bytes MTU (Maximum Transmission Unit). They can significantly improve network performance when transferring large amounts of data by reducing the overhead of packet headers and the number of packets that need to be processed.


**We recommend an MTU of 9000 bytes on all interfaces involved in the data path.** You can check the current MTU of your interfaces:

=== "tune_system.py"

    === "Debian installation"

        ```bash
        sudo /opt/nvidia/holoscan/bin/tune_system.py --check mtu
        ```

    === "From source"

        ```bash
        cd holohub
        sudo ./operators/advanced_network/python/tune_system.py --check mtu
        ```

    ??? abstract "See an example output"

        ```
        2025-03-06 16:51:19 - INFO - Interface eth0 has an acceptable MTU of 9000 bytes.
        2025-03-06 16:51:19 - INFO - Interface eth1 has an acceptable MTU of 9000 bytes.
        ```

=== "manual"

    For a given `if_name` interface:

    ```bash
    if_name=eth0
    ip link show dev $if_name | grep -oE "mtu [0-9]+"
    ```

    ??? abstract "See an example output"

        ```
        mtu 1500
        ```

You can set the MTU for each interface like so, for a given `if_name` name identified [above](#23-configure-the-ip-addresses-of-the-nic-ports):

=== "One-time"

    ```bash
    sudo ip link set dev $if_name mtu 9000
    ```

=== "Persistent"

    === "NetworkManager"

        ```bash
        sudo nmcli connection modify $if_name ipv4.mtu 9000
        sudo nmcli connection up $if_name
        ```

    === "systemd-networkd"

        Assuming you've set an IP address for the interface [above](#23-configure-the-ip-addresses-of-the-nic-ports), you can add the MTU to the interface's network configuration file like so:

        ```bash
        sudo sed -i '/\[Network\]/a MTU=9000' /etc/systemd/network/20-$if_name.network
        sudo systemctl restart systemd-networkd
        ```

??? info "Can I do more than 9000?"

    While your NIC might have a maximum MTU capability larger than 9000, we typically recommend setting the MTU to 9000 bytes, as that is the standard size for jumbo frames that's widely supported for compatibility with other network equipment. When using jumbo frames, all devices in the communication path must support the same MTU size. If any device in between has a smaller MTU, packets will be fragmented or dropped, potentially degrading performance.

    Example with the CX-7 NIC:

    ```bash
    $ ip -d link show dev $if_name | grep -oE "maxmtu [0-9]+"
    maxmtu 9978
    ```

## 4. Running a test application

Holoscan Networking provides a benchmarking application named `adv_networking_bench` that can be used to test the performance of the networking configuration. In this section, we'll walk you through the steps needed to configure the application for your NIC for Tx and Rx, and run a loopback test between the two interfaces with a [physical SFP cable](https://www.nvidia.com/en-us/networking/interconnect/) connecting them.

Make sure to install [`holoscan-networking`](#1-installing-holoscan-networking) beforehand.

### 4.1 Update the loopback configuration

##### Find the application files

Identify the location of the `adv_networking_bench` executable, and of the configuration file named `adv_networking_bench_default_tx_rx.yaml`, for your installation:

=== "Debian installation"

    Both located under `/opt/nvidia/holoscan/examples/adv_networking_bench/`:

    ```bash hl_lines="2 5"
    ls -1 /opt/nvidia/holoscan/examples/adv_networking_bench/
    adv_networking_bench
    adv_networking_bench_default_rx_multi_q.yaml
    adv_networking_bench_default_tx_rx_hds.yaml
    adv_networking_bench_default_tx_rx.yaml
    adv_networking_bench_gpunetio_tx_rx.yaml
    adv_networking_bench_rmax_rx.yaml
    CMakeLists.txt
    default_bench_op_rx.h
    default_bench_op_tx.h
    doca_bench_op_rx.h
    doca_bench_op_tx.h
    kernels.cu
    kernels.cuh
    main.cpp
    ```

=== "From source"

    Both located under `./install/examples/adv_networking_bench/`

    ```bash hl_lines="2 5"
    ls -1 ./install/examples/adv_networking_bench
    adv_networking_bench
    adv_networking_bench_default_rx_multi_q.yaml
    adv_networking_bench_default_tx_rx_hds.yaml
    adv_networking_bench_default_tx_rx.yaml
    adv_networking_bench_gpunetio_tx_rx.yaml
    adv_networking_bench.py
    adv_networking_bench_rmax_rx.yaml
    CMakeLists.txt
    default_bench_op_rx.h
    default_bench_op_tx.h
    doca_bench_op_rx.h
    doca_bench_op_tx.h
    kernels.cu
    kernels.cuh
    main.cpp
    ```

    !!! warning

        The configuration file is also located alongide the application source code at `applications/adv_networking_bench/adv_networking_bench_default_tx_rx.yaml`.
        However, modifying this file will not affect the configuration used by the application executable without rebuilding the application.

        For this reason, we recommend using the configuration file located in the install tree.

!!! note

    The fields in this `yaml` file will be explained in more details in [a section below](#51-understand-the-configuration-parameters). For now, we'll stick to modifying the strict minimum required fields to run the application as-is on your system.

##### Identify your NIC's PCIe addresses

Retrieve the PCIe addresses of both ports of your NIC. We'll arbitrarily use the first for Tx and the second for Rx here:

=== "ibdev2netdev"

    ```bash
    sudo ibdev2netdev -v | awk '{print $1}'
    ```

=== "lspci"

    ```bash
    # `0200` is the PCI-SIG class code for NICs
    # `15b3` is the Vendor ID for Mellanox
    lspci -n | awk '$2 == "0200:" && $3 ~ /^15b3:/ {print $1}'
    ```

??? abstract "See an example output"

    ```
    0005:03:00.0
    0005:03:00.1
    ```

##### Configure the NIC for Tx and Rx

Set the NIC addresses in the `interfaces` section of the `advanced_network` section, making sure to remove the template brackets `< >`. This configures your NIC independently of your application:

- Set the `address` field of the `tx_port` interface to one of these addresses. That interface will be able to transmit ethernet packets.
- Set the `address` field of the `rx_port` interface to the other address. This interface will be able to receive ethernet packets.

```yaml hl_lines="3 7"
interfaces:
    - name: "tx_port"
    address: <0000:00:00.0>       # The BUS address of the interface doing Tx
    tx:
        ...
    - name: "rx_port"
    address: <0000:00:00.0>       # The BUS address of the interface doing Rx
    rx:
        ...
```

???+ abstract "See an example yaml"

    ```yaml hl_lines="3 7"
    interfaces:
        - name: "tx_port"
        address: 0005:03:00.0       # The BUS address of the interface doing Tx
        tx:
            ...
        - name: "rx_port"
        address: 0005:03:00.1       # The BUS address of the interface doing Rx
        rx:
            ...
    ```

##### Configure the application

To run the benchmarking application to run a loopback on your system, you'll need to modify the `bench_tx` section which configures the application itself, to create the packet headers and direct the packets to the NIC. Make sure to remove the template brackets `< >`.

-  `eth_dst_addr` with the MAC address (and not the PCIe address) of the NIC interface you want to use for Rx. You can get the MAC address of your `if_name` interface with `#!bash cat /sys/class/net/$if_name/address`:

```yaml hl_lines="4"
bench_tx:
    interface_name: "tx_port" # Name of the TX port from the advanced_network config
    ...
    eth_dst_addr: <00:00:00:00:00:00> # Destination MAC address - required when Rx flow_isolation=true
    ...
```

???+ abstract "See an example yaml"

    ```yaml hl_lines="4"
    bench_tx:
        interface_name: "tx_port" # Name of the TX port from the advanced_network config
        ...
        eth_dst_addr: 48:b0:2d:ee:83:ad # Destination MAC address - required when Rx flow_isolation=true
        ...
    ```

??? info "Show explanation"

    - `eth_dst_addr` - the destination ethernet MAC address - will be embedded in the packet headers by the application. This is required here because the Rx interface above has `flow_isolation: true` (explained in more details below). In that configuration, only the packets listing the adequate destination MAC address will be accepted by the Rx interface.
    - We ignore the IP fields (`ip_src_addr`, `ip_dst_addr`) for now, as we are testing on a layer 2 network by just connecting a cable between the two interfaces on our system, therefore having mock values has no impact.
    - You might have noted the lack of a `eth_src_addr` field in this `bench_tx` section. This is because the source Ethernet MAC address can be inferred automatically by the Advanced Network library from the PCIe address of the Tx interface referenced above.

### 4.2 Run the loopback test

After having modified the configuration file, ensure you have connected an SFP cable between the two interfaces of your NIC, then run the application with the command below:

=== "Debian installation"

    ```bash
    sudo /opt/nvidia/holoscan/examples/adv_networking_bench/adv_networking_bench adv_networking_bench_default_tx_rx.yaml
    ```

=== "From source"

    === "Bare Metal"

        This assumes you have the required dependencies (holoscan, doca, etc.) installed locally on your system.

        ```bash
        sudo ./install/examples/adv_networking_bench/adv_networking_bench adv_networking_bench_default_tx_rx.yaml
        ```

    === "Containerized"

        ```bash
        ./holohub run-container \
          --img holohub:adv_networking_bench \
          --docker-opts "-u 0 --privileged" \
          -- bash -c "./install/examples/adv_networking_bench/adv_networking_bench adv_networking_bench_default_tx_rx.yaml"
        ```


The application will run indefinitely. You can stop it gracefully with `Ctrl-C`. You can also uncomment and set the `max_duration_ms` field in the `scheduler` section of the configuration file to limit the duration of the run automatically.

??? abstract "See an example output"

    ```log
    [info] [fragment.cpp:599] Loading extensions from configs...
    [info] [gxf_executor.cpp:264] Creating context
    [info] [main.cpp:35] Initializing advanced network operator
    [info] [main.cpp:40] Using ANO manager dpdk
    [info] [adv_network_rx.cpp:35] Adding output port bench_rx_out
    [info] [adv_network_rx.cpp:51] AdvNetworkOpRx::initialize()
    [info] [adv_network_common.h:607] Finished reading advanced network operator config
    [info] [adv_network_dpdk_mgr.cpp:373] Attempting to use 2 ports for high-speed network
    [info] [adv_network_dpdk_mgr.cpp:382] Setting DPDK log level to: Info
    [info] [adv_network_dpdk_mgr.cpp:402] DPDK EAL arguments: adv_net_operator --file-prefix=nwlrbbmqbh -l 3,11,9 --log-level=9 --log-level=pmd.net.mlx5:info -a 0005:03:00.0,txq_inline_max=0,dv_flow_en=2 -a 0005:03:00.1,txq_inline_max=0,dv_flow_en=2
    Log level 9 higher than maximum (8)
    EAL: Detected CPU lcores: 12
    EAL: Detected NUMA nodes: 1
    EAL: Detected shared linkage of DPDK
    EAL: Multi-process socket /var/run/dpdk/nwlrbbmqbh/mp_socket
    EAL: Selected IOVA mode 'VA'
    EAL: 1 hugepages of size 1073741824 reserved, but no mounted hugetlbfs found for that size
    EAL: Probe PCI driver: mlx5_pci (15b3:1021) device: 0005:03:00.0 (socket -1)
    mlx5_net: PCI information matches for device "mlx5_0"
    mlx5_net: enhanced MPS is enabled
    mlx5_net: port 0 MAC address is 48:B0:2D:EE:83:AC
    EAL: Probe PCI driver: mlx5_pci (15b3:1021) device: 0005:03:00.1 (socket -1)
    mlx5_net: PCI information matches for device "mlx5_1"
    mlx5_net: enhanced MPS is enabled
    mlx5_net: port 1 MAC address is 48:B0:2D:EE:83:AD
    TELEMETRY: No legacy callbacks, legacy socket not created
    [info] [adv_network_dpdk_mgr.cpp:298] Port 0 has no RX queues. Creating dummy queue.
    [info] [adv_network_dpdk_mgr.cpp:165] Adjusting buffer size to 9228 for headroom
    [info] [adv_network_dpdk_mgr.cpp:165] Adjusting buffer size to 9128 for headroom
    [info] [adv_network_dpdk_mgr.cpp:165] Adjusting buffer size to 9128 for headroom
    [info] [adv_network_mgr.cpp:116] Registering memory regions
    [info] [adv_network_mgr.cpp:178] Successfully allocated memory region MR_Unused_P0 at 0x100fa0000 type 2 with 9100 bytes (32768 elements @ 9228 bytes total 302383104)
    [info] [adv_network_mgr.cpp:178] Successfully allocated memory region Data_RX_GPU at 0xffff4fc00000 type 3 with 9000 bytes (51200 elements @ 9128 bytes total 467402752)
    [info] [adv_network_mgr.cpp:178] Successfully allocated memory region Data_TX_GPU at 0xffff33e00000 type 3 with 9000 bytes (51200 elements @ 9128 bytes total 467402752)
    [info] [adv_network_mgr.cpp:191] Finished allocating memory regions
    [info] [adv_network_dpdk_mgr.cpp:223] Successfully registered external memory for Data_TX_GPU
    [info] [adv_network_dpdk_mgr.cpp:223] Successfully registered external memory for Data_RX_GPU
    [info] [adv_network_dpdk_mgr.cpp:193] Mapped external memory descriptor for 0xffff4fc00000 to device 0
    [info] [adv_network_dpdk_mgr.cpp:193] Mapped external memory descriptor for 0xffff33e00000 to device 0
    [info] [adv_network_dpdk_mgr.cpp:193] Mapped external memory descriptor for 0xffff4fc00000 to device 1
    [info] [adv_network_dpdk_mgr.cpp:193] Mapped external memory descriptor for 0xffff33e00000 to device 1
    [info] [adv_network_dpdk_mgr.cpp:454] DPDK init (0005:03:00.0) -- RX: ENABLED TX: ENABLED
    [info] [adv_network_dpdk_mgr.cpp:464] Configuring RX queue: UNUSED_P0_Q0 (0) on port 0
    [info] [adv_network_dpdk_mgr.cpp:513] Created mempool RXP_P0_Q0_MR0 : mbufs=32768 elsize=9228 ptr=0x10041c380
    [info] [adv_network_dpdk_mgr.cpp:523] Max packet size needed for RX: 9100
    [info] [adv_network_dpdk_mgr.cpp:564] Configuring TX queue: ADC Samples (0) on port 0
    [info] [adv_network_dpdk_mgr.cpp:607] Created mempool TXP_P0_Q0_MR0 : mbufs=51200 elsize=9000 ptr=0x100c1fc00
    [info] [adv_network_dpdk_mgr.cpp:621] Max packet size needed with TX: 9100
    [info] [adv_network_dpdk_mgr.cpp:632] Setting port config for port 0 mtu:9082
    [info] [adv_network_dpdk_mgr.cpp:663] Initializing port 0 with 1 RX queues and 1 TX queues...
    mlx5_net: port 0 Tx queues number update: 0 -> 1
    mlx5_net: port 0 Rx queues number update: 0 -> 1
    [info] [adv_network_dpdk_mgr.cpp:679] Successfully configured ethdev
    [info] [adv_network_dpdk_mgr.cpp:689] Successfully set descriptors to 8192/8192
    [info] [adv_network_dpdk_mgr.cpp:704] Port 0 not in isolation mode
    [info] [adv_network_dpdk_mgr.cpp:713] Setting up port:0, queue:0, Num scatter:1 pool:0x10041c380
    [info] [adv_network_dpdk_mgr.cpp:734] Successfully setup RX port 0 queue 0
    [info] [adv_network_dpdk_mgr.cpp:756] Successfully set up TX queue 0/0
    [info] [adv_network_dpdk_mgr.cpp:761] Enabling promiscuous mode for port 0
    mlx5_net: [mlx5dr_cmd_query_caps]: Failed to query wire port regc value
    mlx5_net: port 0 Rx queues number update: 1 -> 1
    [info] [adv_network_dpdk_mgr.cpp:775] Successfully started port 0
    [info] [adv_network_dpdk_mgr.cpp:778] Port 0, MAC address: 48:B0:2D:EE:83:AC
    [info] [adv_network_dpdk_mgr.cpp:1111] Applying tx_eth_src offload for port 0
    [info] [adv_network_dpdk_mgr.cpp:454] DPDK init (0005:03:00.1) -- RX: ENABLED TX: DISABLED
    [info] [adv_network_dpdk_mgr.cpp:464] Configuring RX queue: Data (0) on port 1
    [info] [adv_network_dpdk_mgr.cpp:513] Created mempool RXP_P1_Q0_MR0 : mbufs=51200 elsize=9128 ptr=0x125a5b940
    [info] [adv_network_dpdk_mgr.cpp:523] Max packet size needed for RX: 9000
    [info] [adv_network_dpdk_mgr.cpp:621] Max packet size needed with TX: 9000
    [info] [adv_network_dpdk_mgr.cpp:632] Setting port config for port 1 mtu:8982
    [info] [adv_network_dpdk_mgr.cpp:663] Initializing port 1 with 1 RX queues and 0 TX queues...
    mlx5_net: port 1 Rx queues number update: 0 -> 1
    [info] [adv_network_dpdk_mgr.cpp:679] Successfully configured ethdev
    [info] [adv_network_dpdk_mgr.cpp:689] Successfully set descriptors to 8192/8192
    [info] [adv_network_dpdk_mgr.cpp:701] Port 1 in isolation mode
    [info] [adv_network_dpdk_mgr.cpp:713] Setting up port:1, queue:0, Num scatter:1 pool:0x125a5b940
    [info] [adv_network_dpdk_mgr.cpp:734] Successfully setup RX port 1 queue 0
    [info] [adv_network_dpdk_mgr.cpp:764] Not enabling promiscuous mode on port 1 since flow isolation is enabled
    mlx5_net: [mlx5dr_cmd_query_caps]: Failed to query wire port regc value
    mlx5_net: port 1 Rx queues number update: 1 -> 1
    [info] [adv_network_dpdk_mgr.cpp:775] Successfully started port 1
    [info] [adv_network_dpdk_mgr.cpp:778] Port 1, MAC address: 48:B0:2D:EE:83:AD
    [info] [adv_network_dpdk_mgr.cpp:790] Adding RX flow ADC Samples
    [info] [adv_network_dpdk_mgr.cpp:998] Adding IPv4 length match for 1050
    [info] [adv_network_dpdk_mgr.cpp:1018] Adding UDP port match for src/dst 4096/4096
    [info] [adv_network_dpdk_mgr.cpp:814] Setting up RX burst pool with 8191 batches of size 81920
    [info] [adv_network_dpdk_mgr.cpp:833] Setting up RX burst pool with 8191 batches of size 20480
    [info] [adv_network_dpdk_mgr.cpp:875] Setting up TX ring TX_RING_P0_Q0
    [info] [adv_network_dpdk_mgr.cpp:901] Setting up TX burst pool TX_BURST_POOL_P0_Q0 with 10240 pointers at 0x125a0d4c0
    [info] [adv_network_dpdk_mgr.cpp:1186] Config validated successfully
    [info] [adv_network_dpdk_mgr.cpp:1199] Starting advanced network workers
    [info] [adv_network_dpdk_mgr.cpp:1278] Flushing packet on port 1
    [info] [adv_network_dpdk_mgr.cpp:1478] Starting RX Core 9, port 1, queue 0, socket 0
    [info] [adv_network_dpdk_mgr.cpp:1268] Done starting workers
    [info] [default_bench_op_tx.h:79] AdvNetworkingBenchDefaultTxOp::initialize()
    [info] [adv_network_dpdk_mgr.cpp:1637] Starting TX Core 11, port 0, queue 0 socket 0 using burst pool 0x125a0d4c0 ring 0x127690740
    [info] [default_bench_op_tx.h:113] Initialized 4 streams and events
    [info] [default_bench_op_tx.h:130] AdvNetworkingBenchDefaultTxOp::initialize() complete
    [info] [default_bench_op_rx.h:67] AdvNetworkingBenchDefaultRxOp::initialize()
    [info] [gxf_executor.cpp:1797] creating input IOSpec named 'burst_in'
    [info] [default_bench_op_rx.h:104] AdvNetworkingBenchDefaultRxOp::initialize() complete
    [info] [adv_network_tx.cpp:46] AdvNetworkOpTx::initialize()
    [info] [gxf_executor.cpp:1797] creating input IOSpec named 'burst_in'
    [info] [adv_network_common.h:607] Finished reading advanced network operator config
    [info] [gxf_executor.cpp:2208] Activating Graph...
    [info] [gxf_executor.cpp:2238] Running Graph...
    [info] [multi_thread_scheduler.cpp:300] MultiThreadScheduler started worker thread [pool name: default_pool, thread uid: 0]
    [info] [multi_thread_scheduler.cpp:300] MultiThreadScheduler started worker thread [pool name: default_pool, thread uid: 1]
    [info] [multi_thread_scheduler.cpp:300] MultiThreadScheduler started worker thread [pool name: default_pool, thread uid: 2]
    [info] [gxf_executor.cpp:2240] Waiting for completion...
    [info] [multi_thread_scheduler.cpp:300] MultiThreadScheduler started worker thread [pool name: default_pool, thread uid: 3]
    [info] [multi_thread_scheduler.cpp:300] MultiThreadScheduler started worker thread [pool name: default_pool, thread uid: 4]
    ^C[info] [multi_thread_scheduler.cpp:636] Stopping multithread scheduler
    [info] [multi_thread_scheduler.cpp:694] Stopping all async jobs
    [info] [multi_thread_scheduler.cpp:218] Dispatcher thread has stopped checking jobs
    [info] [multi_thread_scheduler.cpp:679] Waiting to join all async threads
    [info] [multi_thread_scheduler.cpp:316] Worker Thread [pool name: default_pool, thread uid: 1] exiting.
    [info] [multi_thread_scheduler.cpp:702] *********************** DISPATCHER EXEC TIME : 476345.364000 ms

    [info] [multi_thread_scheduler.cpp:316] Worker Thread [pool name: default_pool, thread uid: 0] exiting.
    [info] [multi_thread_scheduler.cpp:316] Worker Thread [pool name: default_pool, thread uid: 3] exiting.
    [info] [multi_thread_scheduler.cpp:371] Event handler thread exiting.
    [info] [multi_thread_scheduler.cpp:703] *********************** DISPATCHER WAIT TIME : 47339.961000 ms

    [info] [multi_thread_scheduler.cpp:704] *********************** DISPATCHER COUNT : 197630449

    [info] [multi_thread_scheduler.cpp:316] Worker Thread [pool name: default_pool, thread uid: 2] exiting.
    [info] [multi_thread_scheduler.cpp:705] *********************** WORKER EXEC TIME : 983902.800000 ms

    [info] [multi_thread_scheduler.cpp:706] *********************** WORKER WAIT TIME : 1634522.159000 ms

    [info] [multi_thread_scheduler.cpp:707] *********************** WORKER COUNT : 11817369

    [info] [multi_thread_scheduler.cpp:316] Worker Thread [pool name: default_pool, thread uid: 4] exiting.
    [info] [multi_thread_scheduler.cpp:688] All async worker threads joined, deactivating all entities
    [info] [adv_network_rx.cpp:46] AdvNetworkOpRx::stop()
    [info] [adv_network_dpdk_mgr.cpp:1928] DPDK ANO shutdown called 2
    [info] [adv_network_tx.cpp:41] AdvNetworkOpTx::stop()
    [info] [adv_network_dpdk_mgr.cpp:1928] DPDK ANO shutdown called 1
    [info] [adv_network_dpdk_mgr.cpp:1133] Port 0:
    [info] [adv_network_dpdk_mgr.cpp:1135]  - Received packets:    0
    [info] [adv_network_dpdk_mgr.cpp:1136]  - Transmit packets:    6005066864
    [info] [adv_network_dpdk_mgr.cpp:1137]  - Received bytes:      0
    [info] [adv_network_dpdk_mgr.cpp:1138]  - Transmit bytes:      6389391347584
    [info] [adv_network_dpdk_mgr.cpp:1139]  - Missed packets:      0
    [info] [adv_network_dpdk_mgr.cpp:1140]  - Errored packets:     0
    [info] [adv_network_dpdk_mgr.cpp:1141]  - RX out of buffers:   0
    [info] [adv_network_dpdk_mgr.cpp:1143]    ** Extended Stats **
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_good_packets:          6005070000
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_good_bytes:            6389394480000
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_q0_packets:            6005070000
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_q0_bytes:              6389394480000
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_multicast_bytes:               9589
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_multicast_packets:             22
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_unicast_bytes:         6389394480000
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_multicast_bytes:               9589
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_unicast_packets:               6005070000
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_multicast_packets:             22
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_phy_packets:           6005070022
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_phy_packets:           24
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_phy_bytes:             6413414769677
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_phy_bytes:             9805
    [info] [adv_network_dpdk_mgr.cpp:1133] Port 1:
    [info] [adv_network_dpdk_mgr.cpp:1135]  - Received packets:    6004323692
    [info] [adv_network_dpdk_mgr.cpp:1136]  - Transmit packets:    0
    [info] [adv_network_dpdk_mgr.cpp:1137]  - Received bytes:      6388600255072
    [info] [adv_network_dpdk_mgr.cpp:1138]  - Transmit bytes:      0
    [info] [adv_network_dpdk_mgr.cpp:1139]  - Missed packets:      746308
    [info] [adv_network_dpdk_mgr.cpp:1140]  - Errored packets:     0
    [info] [adv_network_dpdk_mgr.cpp:1141]  - RX out of buffers:   5047027287
    [info] [adv_network_dpdk_mgr.cpp:1143]    ** Extended Stats **
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_good_packets:          6004323692
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_good_bytes:            6388600255072
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_missed_errors:         746308
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_mbuf_allocation_errors:                5047027287
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_q0_packets:            6004323692
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_q0_bytes:              6388600255072
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_q0_errors:             5047027287
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_unicast_bytes:         6389394480000
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_multicast_bytes:               9589
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_unicast_packets:               6005070000
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_multicast_packets:             22
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_multicast_bytes:               9589
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_multicast_packets:             22
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_phy_packets:           24
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_phy_packets:           6005070022
    [info] [adv_network_dpdk_mgr.cpp:1173]       tx_phy_bytes:             9805
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_phy_bytes:             6413414769677
    [info] [adv_network_dpdk_mgr.cpp:1173]       rx_out_of_buffer:         746308
    [info] [adv_network_dpdk_mgr.cpp:1935] ANO DPDK manager shutting down
    [info] [adv_network_dpdk_mgr.cpp:1622] Total packets received by application (port/queue 1/0): 6004323692
    [info] [adv_network_dpdk_mgr.cpp:1698] Total packets transmitted by application (port/queue 0/0): 6005070000
    [info] [multi_thread_scheduler.cpp:645] Multithread scheduler stopped.
    [info] [multi_thread_scheduler.cpp:664] Multithread scheduler finished.
    [info] [gxf_executor.cpp:2243] Deactivating Graph...
    [info] [multi_thread_scheduler.cpp:491] TOTAL EXECUTION TIME OF SCHEDULER : 523694.460857 ms

    [info] [gxf_executor.cpp:2251] Graph execution finished.
    [info] [adv_network_dpdk_mgr.cpp:1928] DPDK ANO shutdown called 0
    [info] [default_bench_op_tx.h:51] ANO benchmark TX op shutting down
    [info] [default_bench_op_rx.h:56] Finished receiver with 6388570603520/6004295680 bytes/packets received and 0 packets dropped
    [info] [default_bench_op_rx.h:61] ANO benchmark RX op shutting down
    [info] [default_bench_op_rx.h:108] AdvNetworkingBenchDefaultRxOp::freeResources() start
    [info] [default_bench_op_rx.h:116] AdvNetworkingBenchDefaultRxOp::freeResources() complete
    [info] [gxf_executor.cpp:294] Destroying context
    ```

To inspect the speed the data is moving through the NIC, run `mlnx_perf` on one of the interfaces in a separate terminal, concurrently with the application running:

```bash
sudo mlnx_perf -i $if_name
```

??? abstract "See an example output"

    On IGX with RTX A6000, we are able to hit close to the 100 Gbps linerate with this configuration:
    ```log
      rx_vport_unicast_packets: 11,614,900
        rx_vport_unicast_bytes: 12,358,253,600 Bps   = 98,866.2 Mbps
                rx_packets_phy: 11,614,847
                  rx_bytes_phy: 12,404,657,664 Bps   = 99,237.26 Mbps
     rx_1024_to_1518_bytes_phy: 11,614,936
                rx_prio0_bytes: 12,404,738,832 Bps   = 99,237.91 Mbps
              rx_prio0_packets: 11,614,923
    ```

??? tip "Troubleshooting"

    ??? failure "EAL: failed to parse device"

        Make sure to set valid PCIe addresses in the `address` fields in `interfaces`, per [instructions above](#configure-the-nic-for-tx-and-rx).

    ??? failure "Invalid MAC address format"

        Make sure to set a valid MAC address in the `eth_dst_addr` field in `bench_tx`, per [instructions above](#configure-the-application).

    ??? failure "mlx5_common: Fail to create MR for address [...] Could not DMA map EXT memory"

        Example error:

        ```log
        mlx5_common: Fail to create MR for address (0xffff2fc00000)
        mlx5_common: Device 0005:03:00.0 unable to DMA map
        [critical] [adv_network_dpdk_mgr.cpp:188] Could not DMA map EXT memory: -1 err=Invalid argument
        [critical] [adv_network_dpdk_mgr.cpp:430] Failed to map MRs
        ```

        [Make sure that `nvidia-peermem` is loaded](#24-enable-gpudirect).

    ??? failure "EAL: Couldn't get fd on hugepage file [..] error allocating rte services array"

        Example error:

        ```log
        EAL: get_seg_fd(): open '/mnt/huge/nwlrbbmqbhmap_0' failed: Permission denied
        EAL: Couldn't get fd on hugepage file
        EAL: error allocating rte services array
        EAL: FATAL: rte_service_init() failed
        EAL: rte_service_init() failed
        ```

        Ensure you run as root, using `sudo`.

    ??? failure "EAL: Cannot get hugepage information."

        ```log
        EAL: x hugepages of size x reserved, no mounted hugetlbfs found for that size
        ```

        Ensure your [hugepages are mounted](#34-enable-huge-pages).

        ```log
        EAL: No free x kB hugepages reported on node 0
        ```

        - Ensure you have [allocated hugepages](#34-enable-huge-pages).
        - If you have already, check if they are any free left with `grep Huge /proc/meminfo`.

            ??? abstract "See an example output"

                No more space here!

                ```
                HugePages_Total:       2
                HugePages_Free:        0
                HugePages_Rsvd:        0
                HugePages_Surp:        0
                Hugepagesize:    1048576 kB
                Hugetlb:         2097152 kB
                ```

        - If not, you can delete dangling hugepages under your hugepage mount point. That happens when your previous application run crashes.

            ```bash
            sudo rm -rf /dev/hugepages/* # default mount point
            sudo rm -rf /mnt/huge/*      # custom mount point
            ```

    ??? failure "Could not allocate x MB of GPU memory [...] Failed to allocate GPU memory"

        Check your GPU utilization:

        ```bash
        nvidia-smi pmon -c 1
        ```

        You might need to kill some of the listed processes to free up GPU VRAM.


## 5. Building your own application

This section will guide you through building your own application using the `adv_networking_bench` as an example. Make sure to install [`holoscan-networking`](#1-installing-holoscan-networking) first.

### 5.1 Understand the configuration parameters

!!! note

    The configuration below will be analyzed in the context of the application consuming it, as defined in the `main.cpp` file. You can look it up when the "sample application code" is referenced.

    === "Debian installation"

        ```bash
        /opt/nvidia/holoscan/examples/adv_networking_bench/main.cpp
        ```

    === "From source"

        ```bash
        ./applications/adv_networking_bench/cpp/main.cpp
        ```

    If you are not yet familiar with how Holoscan applications are constructed, please refer to the [Holoscan SDK documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_core.html) first.

Let's look at the `adv_networking_bench_default_tx_rx.yaml` file below. Click on the (1) icons below to expand explanations for each annotated line.
{ .annotate }

1. The cake is a lie :cake:

```yaml
scheduler: # (1)!
  check_recession_period_ms: 0
  worker_thread_number: 5
  stop_on_deadlock: true
  stop_on_deadlock_timeout: 500
  # max_duration_ms: 20000

advanced_network: # (2)!
  cfg:
    version: 1
    manager: "dpdk" # (3)!
    master_core: 3 # (4)!
    debug: false
    log_level: "info"

    memory_regions: # (5)!
    - name: "Data_TX_GPU" # (6)!
      kind: "device" # (7)!
      affinity: 0 # (8)!
      num_bufs: 51200 # (9)!
      buf_size: 1064 # (10)!
    - name: "Data_RX_GPU"
      kind: "device"
      affinity: 0
      num_bufs: 51200
      buf_size: 1000
    - name: "Data_RX_CPU"
      kind: "huge"
      affinity: 0
      num_bufs: 51200
      buf_size: 64

    interfaces: # (11)!
    - name: "tx_port" # (12)!
      address: <0000:00:00.0> # (13)! # The BUS address of the interface doing Tx
      tx: # (14)!
        queues: # (15)!
        - name: "tx_q_0" # (16)!
          id: 0 # (17)!
          batch_size: 10240 # (18)!
          cpu_core: 11 # (19)!
          memory_regions: # (20)!
            - "Data_TX_GPU"
          offloads: # (21)!
            - "tx_eth_src"
    - name: "rx_port"
      address: <0000:00:00.0> # (22)! # The BUS address of the interface doing Rx
      rx:
        flow_isolation: true # (23)!
        queues:
        - name: "rx_q_0"
          id: 0
          cpu_core: 9
          batch_size: 10240
          memory_regions: # (24)!
            - "Data_RX_CPU"
            - "Data_RX_GPU"
        flows: # (25)!
        - name: "flow_0" # (26)!
          id: 0 # (27)!
          action: # (28)!
            type: queue
            id: 0
          match: # (29)!
            udp_src: 4096
            udp_dst: 4096
            ipv4_len: 1050

bench_rx: # (30)!
  interface_name: "rx_port" # Name of the RX port from the advanced_network config
  gpu_direct: true          # Set to true if using a GPU region for the Rx queues.
  split_boundary: true      # Whether header and data are split for Rx (Header to CPU)
  batch_size: 10240
  max_packet_size: 1064
  header_size: 64

bench_tx: # (31)!
  interface_name: "tx_port" # Name of the TX port from the advanced_network config
  gpu_direct: true          # Set to true if using a GPU region for the Tx queues.
  split_boundary: 0         # Byte boundary where header and data are split for Tx, 0 if no split
  batch_size: 10240
  payload_size: 1000
  header_size: 64
  eth_dst_addr: <00:00:00:00:00:00> # Destination MAC address - required when Rx flow_isolation=true
  ip_src_addr: <1.2.3.4>    # Source IP address - required on layer 3 network
  ip_dst_addr: <5.6.7.8>    # Destination IP address - required on layer 3 network
  udp_src_port: 4096        # UDP source port
  udp_dst_port: 4096        # UDP destination port
```

1. The `scheduler` section is passed to the multi threaded scheduler we declare in the `#!cpp main()` function of this application. See the [holoscan SDK documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/components/schedulers.html) and [API docs](https://docs.nvidia.com/holoscan/sdk-user-guide/api/cpp/classholoscan_1_1multithreadscheduler.html) for more details. This is related to the Holoscan core library and is not specific to Holoscan Networking.
2. The `advanced_network` section is passed to the `advanced_network::adv_net_init` which is responsible for setting up the NIC. That function should be called in your `#!cpp Application::compose()` function.
3. `manager` is the backend networking library. default: `dpdk`. Other: `gpunetio` (DOCA GPUNet IO + DOCA Ethernet & Flow). Coming soon: `rivermax`, `rdma`.
4. `master_core` is the ID of the CPU core used for setup. It does not need to be isolated, and is recommended to differ differ from the `cpu_core` fields below used for polling the NIC.
5. The `memory_regions` section lists where the NIC will write/read data from/to when bypassing the OS kernel. Tip: when using GPU buffer regions, keeping the sum of their buffer sizes lower than 80% of your BAR1 size is generally a good rule of thumb 👍.
6. A descriptive name for that memory region to refer to later in the `interfaces` section.
7. The type of memory region. Best options are `device` (GPU), or `huge` (pages - CPU). Also supported but not recommended are `malloc` (CPU) and `pinned` (CPU).
8. The GPU ID for `device` memory regions. The NUMA node ID for CPU memory regions.
9. The number of buffers in the memory region. A higher value means more time to process the data, but it takes additional space on the GPU BAR1. Too low increases the risk of dropping packets from the NIC having nowhere to write (Rx) or the risk of higher latency from buffering (Tx). Need a rule of thumb 👍? 5x the `batch_size` below is a good starting point.
10. The size of each buffer in the memory region. These should be equal to your maximum packet size, or less if breaking down packets (ex: header data split, see the `rx` queue below).
11. The `interfaces` section lists the NIC interfaces that will be configured for the application.
12. A descriptive name for that interface, currently only used for logging.
13. The PCIe/bus address of that interface, as identified in previous sections.
14. Each interface can have a `tx` (transmitting) or `rx` (receiving) section, or both if you'd like to configure both Tx and Rx on the same interface.
15. The `queues` section lists the queues for that interface. Queues are a core concept of NICs: they handle the actual receiving or transmitting of network packets. Rx queues buffer incoming packets until they can be processed by the application, while Tx queues hold outgoing packets waiting to be sent on the network. The simplest setup uses only one receive and one transmit queue. Using more queues allows multiple streams of network traffic to be processed in parallel, as each queue can be assigned to a specific CPU core, and are assigned their own memory regions that are not shared.
16. A descriptive name for that queue, currently only used for logging.
17. The ID of that queue, which can be referred to later in the `flows` section.
18. The number of packets per batch (or burst). Your Rx operator will have access to packets from the NIC when it receives enough packets for a whole batch/burst. Your Tx operator needs to ensure it does not send more packets than this value on each `#!cpp Operator::compute()` call.
19. The ID of the CPU core that this queue will use to poll the NIC. Ideally one [isolated core](#35-isolate-cpu-cores) per queue.
20. The list of memory regions where this queue will write/read packets from/to. The order matters: the first memory region will be used first to read/write from until it fills up one buffer (`buf_size`), after which it will move to the next region in the list and so on until the packet is fully written/read. See the `memory_regions` for the `rx` queue below for an example.
21. The `offloads` section (Tx queues only) lists optional tasks that can be offloaded to the NIC. The only value currently supported is `tx_eth_src`, that lets the NIC insert the ethernet source mac address in the packet headers. Note: IP, UDP, and Ethernet Checksums or CRC are always done by the NIC currently and are not optional.
22. Same as for `tx_port`. Each interface in this list should have a unique mac address. This one will do `rx` per config below.
23. Whether to isolate the Rx flow. If true, any incoming packets that does not match the MAC address of this interface - or isn't directed to a queue when the `flows` section below is used - will be delegated back to Linux for processing (no kernel bypass). This is useful to let this interface handle ARP, ICMP, etc. Otherwise, any packets sent to this interface (ex: ping) will need to be processed (or dropped) by your application.
24. This scenario is called HDS (Header-Data Split): the packet will first be written to a buffer in the `Data_RX_CPU` memory region, filling its `buf_size` of 64 bytes - which is consistent with the size of our header - then the rest of the packet will be written to the `Data_RX_GPU` memory region. Its `buf_size` of 1000 bytes is just what we need to write the payload size for our application, no byte wasted!
25. The list of flows. Flows are responsible for routing packets to the correct queue based on various properties. If this field is missing, all packets will be routed to the first queue.
26. The flow name, currently only used for logging.
27. The flow `id` is used to tag the packets with what flow it arrived on. This is useful when sending multiple flows to a single queue, as the user application can differentiate which flow (i.e. rules) matched the packet based on this ID.
28. What to do with packets that match this flow. The only supported action currently is `type: queue` to send the packet to a queue given its `id`.
29. List of rules to match packets against. All rules must be met for a packet to match the flow. Currently supported rules include `udp_src` and `udp_dst` (port numbers), `ipv4_len` (#TODO#) etc.
30. The `bench_rx` section is passed to the `AdvNetworkingBenchDefaultRxOp` operator in the `#!cpp Application::compose()` function of the sample application. This operator is a custom operator implemented in `default_bench_op_rx.h` that pulls and aggregates packets received from the NIC, with parameters specific to its own implementation, which can be used as a reference for your own Rx operator. The first parameter, `interface_name`, is used to specify which NIC interface to use for the Rx operation. The following parameters are should align with how `memory_regions` and `queues` were configured for the `rx` interface.
31. The `bench_tx` section is passed to the `AdvNetworkingBenchDefaultTxOp` operator in the `#!cpp Application::compose()` function of the sample application. This operator is a custom operator implemented in `default_bench_op_tx.h` that generates dummy packets to send to the NIC, with parameters specific to its own implementation, which can be used as a reference for your own Tx operator. The first parameter, `interface_name`, is used to specify which NIC interface to use for the Tx operation. The following parameters up to `header_size` should align with how `memory_regions` and `queues` were configured for the `tx` interface. The remaining parameters are used to fill-in the ethernet header of the packets (ETH, IP, UDP).

### 5.2 Create your own Rx operator

!!! example "Under construction"

    This section is under construction. Refer to the implementation of the `AdvNetworkingBenchDefaultRxOp` for an example.

    === "Debian installation"

        ```bash
        /opt/nvidia/holoscan/examples/adv_networking_bench/default_bench_op_rx.h
        ```

    === "From source"

        ```bash
        ./applications/adv_networking_bench/cpp/default_bench_op_rx.h
        ```

!!! note

    Design investigations are expected soon for a generic packet aggregator operator.

### 5.3 Create your own Tx operator

!!! example "Under construction"

    This section is under construction. Refer to the implementation of the `AdvNetworkingBenchDefaultTxOp` for an example.

    === "Debian installation"

        ```bash
        /opt/nvidia/holoscan/examples/adv_networking_bench/default_bench_op_tx.h
        ```

    === "From source"

        ```bash
        ./applications/adv_networking_bench/cpp/default_bench_op_tx.h
        ```

!!! note

    Designs investigations are expected soon for a generic way to prepare packets to send to the NIC.

### 5.4 Build with CMake

=== "Debian installation"

    1. Create a source directory and write your source file(s) for your application and custom operators.
    2. Create a `CMakeLists.txt` file in your source directory like this one:

        ```cmake
        cmake_minimum_required(VERSION 3.20)
        project(my_app CXX) # Add CUDA if writing .cu kernels

        find_package(holoscan 2.6 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")
        find_package(holoscan-networking REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")

        # Create an executable
        add_executable(my_app
            my_app.cpp
            ...
        )
        target_include_directories(my_app
            PRIVATE
                my_include_dirs/
                ...
        )
        target_link_libraries(my_app
            PRIVATE
                holoscan::core
                holoscan::ops::advanced_network_rx
                holoscan::ops::advanced_network_tx
                my_other_dependencies
                ...
        )

        # Copy the config file to the build directory for convenience referring to it
        add_custom_target(my_app_config_yaml
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/my_app_config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/my_app_config.yaml"
        )
        add_dependencies(my_app my_app_config_yaml)
        ```

    3. Build your application like so:

        ```bash
        # Your chosen paths
        src_dir="."
        build_dir="build"

        # Configure the build
        cmake -S "$src_dir" -B "$build_dir"

        # Build the application
        cmake --build "$build_dir" -j
        ```

        ??? failure "Failed to detect a default CUDA architecture."

            Add the path to your installation of `nvcc` to your `PATH`, or pass its to the cmake configuration command like so (adjust to your CUDA/nvcc installation path):

            ```bash
            cmake -S "$src_dir" -B "$build_dir" -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
            ```

    4. Run your application like so:

        ```bash
        "./$build_dir/my_app my_app_config.yaml"
        ```

=== "From source"

    1. Create an application directory under [`applications/`](https://github.com/nvidia-holoscan/holohub/tree/main/applications) in your clone of the HoloHub repository, and write your source file(s) for your application and custom operators.
    2. Add the following to the [`application/CMakeLists.txt`](https://github.com/nvidia-holoscan/holohub/blob/main/applications/adv_networking_bench/CMakeLists.txt) file:

        ```cmake
        add_holohub_application(my_app DEPENDS OPERATORS advanced_network)
        ```

    3. Create a `CMakeLists.txt` file in your application directory like this one:

        ```cmake
        cmake_minimum_required(VERSION 3.20)
        project(my_app CXX) # Add CUDA if writing .cu kernels

        find_package(holoscan 2.6 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")

        # Create an executable
        add_executable(my_app
            my_app.cpp
            ...
        )
        target_include_directories(my_app
            PRIVATE
                my_include_dirs/
                ...
        )
        target_link_libraries(my_app
            PRIVATE
                holoscan::core
                holoscan::ops::advanced_network_rx
                holoscan::ops::advanced_network_tx
                my_other_dependencies
                ...
        )

        # Copy the config file to the build directory for convenience referring to it
        add_custom_target(my_app_config_yaml
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/my_app_config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/my_app_config.yaml"
        )
        add_dependencies(my_app my_app_config_yaml)
        ```

    4. Build your application like so:

        ```bash
        ./holohub build my_app
        ```

    5. Run your application like so:

        ```bash
        ./holohub run --img holohub:my_app --docker-opts "-u 0 --privileged" --bash -c "./build/my_app/applications/my_app my_app_config.yaml"
        ```

        or, if you have set up a shortcut to run your application with its config file through its `metadata.json` (see other apps for examples):

        ```bash
        ./holohub run --no-local-build --container_args " -u 0 --privileged"
        ```
