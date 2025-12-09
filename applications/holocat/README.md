# HoloCat - EtherCAT Real-time Integration

![HoloCat Logo](docs/holocat_logo.png)

HoloCat is a high-performance EtherCAT master application that integrates the acontis EC-Master SDK with NVIDIA's Holoscan platform for real-time industrial automation and robotics applications.

## Overview

HoloCat provides deterministic, microsecond-level EtherCAT communication capabilities within the Holoscan ecosystem, enabling:

- **Real-time Control**: Sub-millisecond cycle times with deterministic latency
- **Industrial Integration**: Support for standard EtherCAT devices and protocols
- **Holoscan Native**: Full integration with Holoscan operators and data flow
- **High Performance**: Optimized for NVIDIA hardware acceleration

## Features

- ✅ **Real-time EtherCAT Master** - Full IEC 61158 EtherCAT specification compliance
- ✅ **Holoscan Integration** - Native operator interface with input/output ports
- ✅ **Process Data I/O** - Direct access to EtherCAT process data
- ✅ **Multiple Link Layers** - Raw sockets, DPDK, hardware-specific drivers
- ✅ **Distributed Clocks** - Precise time synchronization across EtherCAT network
- ✅ **Configuration Management** - YAML-based configuration with ENI file support
- ✅ **Container Support** - Docker containerization with capability management

## Prerequisites

### Required Dependencies

1. **acontis EC-Master SDK** (Commercial License)
   - Version 3.2.3 or later
   - Download from: https://www.acontis.com/en/ethercat-master.html
   - License: Commercial (contact acontis for licensing)

2. **NVIDIA Holoscan SDK**
   - Version 3.0.0 or later
   - Included in HoloHub environment

3. **System Requirements**
   - Linux x86_64 or aarch64
   - Network interface capable of EtherCAT communication
   - Root privileges or appropriate capabilities

### Optional (Recommended)

- **Real-time Kernel** - Linux with PREEMPT_RT patches for best performance
- **Dedicated Network Interface** - Separate Ethernet adapter for EtherCAT

## Installation

### Prerequisites
```bash
# Set EC-Master SDK path
export ECMASTER_ROOT=/home/hking/devel/ethercat/ecm

# Verify installation (optional)
./applications/holocat/scripts/verify_ecmaster.sh
```

### Build
```bash
# Navigate to HoloHub directory
cd /path/to/holohub

# Build using HoloHub CLI (recommended)
./holohub build holocat --local

# Alternative: Direct CMake build
cmake -B build/holocat -S applications/holocat
cmake --build build/holocat
```

### Run
```bash
# Show help and version
./build/holocat/applications/holocat/cpp/holocat --help
./build/holocat/applications/holocat/cpp/holocat --version

# Print configuration (requires valid config file)
./build/holocat/applications/holocat/cpp/holocat --config config.yaml --print-config

# Run with configuration file
./build/holocat/applications/holocat/cpp/holocat --config config.yaml
```

## Configuration

### Basic Configuration

Create `holocat_config.yaml`:

```yaml
holocat:
  # Network adapter for EtherCAT
  adapter_name: "eth0"  # Change to your EtherCAT interface
  
  # EtherCAT configuration file
  eni_file: "/tmp/holocat_config.xml"
  
  # Cycle time in microseconds
  cycle_time_us: 1000  # 1ms cycle time
  
  # Real-time priorities (1-99)
  rt_priority: 39
  job_thread_priority: 98
  
  # Enable real-time scheduling
  enable_rt: true

# Holoscan application configuration
holoscan:
  logging:
    level: "info"
```

### Network Interface Setup

```bash
# Find available network interfaces
ip link show

# Configure interface for EtherCAT (example)
sudo ip link set eth0 up
sudo ethtool -s eth0 speed 100 duplex full autoneg off
```

### ENI File Generation

Use EtherCAT configuration tools to generate your ENI file:
- TwinCAT System Manager
- acontis EC-Engineer
- Other EtherCAT configuration tools

## Usage

### Command Line Options

```bash
# Show help and version
./build/holocat/applications/holocat/cpp/holocat --help
./build/holocat/applications/holocat/cpp/holocat --version

# Print configuration (requires valid config file)
./build/holocat/applications/holocat/cpp/holocat --config config.yaml --print-config

# Run with configuration file
./build/holocat/applications/holocat/cpp/holocat --config config.yaml
```


## Configuration File Format

Create a YAML file with the following structure:

```yaml
holocat:
  # EtherCAT network adapter name
  adapter_name: "enx6c6e071e9c45"
  
  # EtherCAT Network Information (ENI) file path
  eni_file: "configs/eni2.xml"
  
  # EtherCAT bus cycle time in microseconds
  cycle_time_us: 1000
  
  # Real-time thread priority (1-99)
  rt_priority: 39
  job_thread_priority: 98
  enable_rt: true
  
  # Process data offsets (hardware-specific)
  dio_out_offset: 80
  dio_in_offset: 144
  
  # Advanced settings
  max_acyc_frames: 32
  job_thread_stack_size: 0x8000

# Holoscan application configuration
holoscan:
  logging:
    level: "info"
```

## Configuration Precedence

Parameters are applied in the following order (highest to lowest priority):

1. **Command line arguments** (highest priority)
2. **Configuration file values**
3. **Built-in defaults** (lowest priority)

## Command Line Options

```bash
# Show help
./holocat --help

# Show version
./holocat --version

# Load configuration file
./holocat --config /path/to/config.yaml

# Print loaded configuration and exit
./holocat --print-config
```

# Run with custom configuration
./holohub run holocat --config /path/to/custom_config.yaml
```

### Container Mode

```bash
# Build container with EC-Master SDK
docker build \
  --mount=type=bind,source=$ECMASTER_ROOT,target=/mnt/ecmaster \
  -f applications/holocat/Dockerfile \
  -t holocat:latest .

# Run container with network access
docker run --rm -it \
  --cap-add=NET_RAW \
  --cap-add=SYS_NICE \
  --cap-add=IPC_LOCK \
  --network=host \
  holocat:latest
```

### Integration with Other Holoscan Operators

```cpp
#include "holocat_operator.hpp"

class MyEtherCATApp : public holoscan::Application {
public:
  void compose() override {
    // Create HoloCat operator
    auto ethercat = make_operator<ops::HolocatOp>("ethercat",
      Arg("adapter_name") = "eth0",
      Arg("eni_file") = "/path/to/config.xml",
      Arg("cycle_time_us") = 1000U
    );
    
    // Create other operators
    auto processor = make_operator<MyProcessorOp>("processor");
    auto visualizer = make_operator<ops::HolovizOp>("viz");
    
    // Connect data flow
    add_flow(ethercat, processor, {{"digital_inputs", "sensor_data"}});
    add_flow(processor, ethercat, {{"control_outputs", "digital_outputs"}});
    add_flow(processor, visualizer, {{"display_data", "receivers"}});
  }
};
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Ensure capabilities are set
   sudo setcap 'cap_net_raw,cap_sys_nice,cap_ipc_lock=ep' ./holocat
   ```

2. **Network Interface Not Found**
   ```bash
   # List available interfaces
   ip link show
   # Update adapter_name in configuration
   ```

3. **EC-Master SDK Not Found**
   ```bash
   # Verify ECMASTER_ROOT environment variable
   echo $ECMASTER_ROOT
   ls -la $ECMASTER_ROOT/SDK/INC/EcMaster.h
   ```

4. **Real-time Performance Issues**
   ```bash
   # Check kernel configuration
   uname -r  # Should show 'rt' for real-time kernel
   
   # Verify CPU isolation (optional)
   cat /proc/cmdline | grep isolcpus
   ```

### Performance Tuning

1. **CPU Isolation**
   ```bash
   # Add to kernel command line
   isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
   ```

2. **Network Interface Tuning**
   ```bash
   # Disable power management
   sudo ethtool -s eth0 wol d
   
   # Set interrupt affinity
   echo 2 > /proc/irq/24/smp_affinity
   ```

3. **Memory Management**
   ```bash
   # Increase locked memory limit
   ulimit -l unlimited
   ```

## Hardware Support

### Tested EtherCAT Devices

- **Wago I/O Modules**: 750-430 (DI), 750-530 (DO), 750-550 (AI), 750-560 (AO)
- **Beckhoff Terminals**: EK1100, EL1008, EL2008, EL3008, EL4008
- **Omron Devices**: NX-series I/O modules
- **Schneider Electric**: TM5 series modules

### Network Adapters

- **Intel**: i210, i350, 82574L, 82599ES
- **Realtek**: RTL8111, RTL8169
- **Broadcom**: NetXtreme series
- **Mellanox**: ConnectX series (with DPDK)

## Development

### Building from Source

```bash
# Clone HoloHub repository
git clone https://github.com/nvidia-holoscan/holohub.git
cd holohub

# Set up EC-Master SDK
export ECMASTER_ROOT=/home/hking/devel/ethercat/ecm

# Verify SDK (optional)
./applications/holocat/scripts/verify_ecmaster.sh

# Build holocat
./holohub build holocat --local
```

### Adding Custom Operators

Extend the `HolocatOp` class or create new operators that interface with EtherCAT:

```cpp
class CustomEtherCATOp : public holoscan::ops::HolocatOp {
public:
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    // Custom EtherCAT processing logic
    HolocatOp::compute(input, output, context);
    
    // Add your custom functionality
  }
};
```

## License

- **HoloCat Application**: Apache 2.0 License
- **EC-Master SDK**: Commercial license (acontis technologies GmbH)
- **Holoscan SDK**: Apache 2.0 License

## Support

- **HoloHub Issues**: https://github.com/nvidia-holoscan/holohub/issues
- **Holoscan Documentation**: https://docs.nvidia.com/holoscan/
- **EC-Master Support**: https://www.acontis.com/en/support.html

## Contributing

Contributions are welcome! Please see the [HoloHub Contributing Guide](../../CONTRIBUTING.md) for details.

## Acknowledgments

- acontis technologies GmbH for the EC-Master SDK
- NVIDIA Holoscan team for the platform integration
- EtherCAT Technology Group for the EtherCAT specification
