# HoloCat - EtherCAT Real-time Integration

![HoloCat Logo](docs/holocat_logo.png)

HoloCat is an EtherCAT master application that integrates the acontis EC-Master SDK with NVIDIA's Holoscan platform.

## Overview

HoloCat provides deterministic EtherCAT communication capabilities within the Holoscan ecosystem, enabling:

- **Real-time Control**
- **Holoscan Native**

## Prerequisites

### Required Dependencies

1. **acontis EC-Master SDK** (Commercial License)
   - Version 3.2.3 or later

## Usage

### Prerequisites
```bash
# Set EC-Master SDK path
export ECMASTER_ROOT=/home/hking/devel/ethercat/ecm

# Verify installation (optional)
./applications/holocat/scripts/verify_ecmaster.sh
```

### Build
```bash
# Build using HoloHub CLI (recommended)
./holohub build holocat --local
```

### Run
```bash
# Run with configuration file
./build/holocat/applications/holocat/cpp/holocat --config ./applications/holocat/configs/holocat_config.yaml
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

### ENI File Generation

Use EtherCAT configuration tools to generate your ENI file.


## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Ensure capabilities are set
   sudo setcap 'cap_net_raw=ep' ./build/holocat/applications/holocat/holocat
   ```

3. **EC-Master SDK Not Found**
   ```bash
   # Verify ECMASTER_ROOT environment variable
   echo $ECMASTER_ROOT
   ls -la $ECMASTER_ROOT/SDK/INC/EcMaster.h
   ```