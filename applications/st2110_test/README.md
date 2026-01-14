# ST2110 Test Application

Simple test application to verify ST 2110-20 video reception using the ST2110SourceOp operator with DPDK backend.

## Purpose

This application tests the ST2110 source operator by:
1. Receiving ST 2110-20 video packets via DPDK
2. Assembling packets into video frames using zero-copy HDS mode
3. Displaying the video stream using Holoviz

## Quick Start

```bash
# 1. Set up virtual environment (first time only)
python3 -m venv venv
./venv/bin/pip install holoscan-cu13

# 2. Configure your network (edit st2110_test_config.yaml)
nano st2110_test_config.yaml  # Update PCI address

# 3. Run the application
./run.sh
```

**See [SETUP.md](SETUP.md) for detailed setup instructions.**

## Prerequisites

### Hardware
- NVIDIA Thor AGX (or compatible system)
- NVIDIA ConnectX-6 or later NIC (or Thor AGX MGBE)
- ST 2110-20 video source streaming to the network

### Software
- ST2110SourceOp operator built (see `../../operators/st2110_source/BUILD.md`)
- DPDK configured and NIC bound to vfio-pci driver
- ST 2110 video source streaming to multicast 239.0.0.1:5004

## Quick Start

### 1. Configure Your Network

Edit `st2110_test_config.yaml` and update:

```yaml
interfaces:
  - name: "st2110_rx"
    address: "0005:03:00.0"  # ← Your NIC's PCI address
```

Find your PCI address:
```bash
lspci | grep -i ethernet
# Example output: 0005:03:00.0 Ethernet controller: NVIDIA Corporation
```

### 2. Bind NIC to DPDK

```bash
# Take interface down
sudo ifconfig mgbe0_0 down

# Bind to DPDK
sudo dpdk-devbind.py --bind=vfio-pci 0005:03:00.0

# Verify
sudo dpdk-devbind.py --status
```

### 3. Start Your ST 2110 Source

Ensure you have a ST 2110-20 video source streaming:
- **Multicast address**: 239.0.0.1
- **UDP port**: 5004
- **Format**: YCbCr-4:2:2 10-bit (or RGBA/RGB)
- **Resolution**: 1920x1080@60fps

### 4. Run the Test Application

```bash
# From holohub root directory
cd applications/st2110_test

# Run with sudo (required for DPDK)
sudo python3 st2110_test_app.py
```

## Expected Behavior

When working correctly:
1. Console shows "ST2110 Test Application" banner
2. Logs indicate DPDK initialization
3. Video window opens showing the ST 2110 stream
4. Console displays packet reception statistics
5. Frame rate should be near 60 FPS

Press **ESC** or **Q** to quit.

## Troubleshooting

### No Video Displayed

**Check ST 2110 source:**
```bash
# Verify multicast traffic is reaching the system
sudo tcpdump -i mgbe0_0 dst 239.0.0.1 and udp port 5004
```

**Check DPDK binding:**
```bash
sudo dpdk-devbind.py --status
# NIC should show "drv=vfio-pci" and not "Active"
```

### "Cannot import ST2110SourceOp" Error

**Build the operator:**
```bash
cd ../../operators/st2110_source
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/opt/nvidia/holoscan/lib;$HOME/Source/holohub/install/lib/cmake" \
         -DCMAKE_BUILD_TYPE=Release
make
```

### DPDK Initialization Fails

**Check huge pages:**
```bash
cat /proc/meminfo | grep Huge
# Should show: HugePages_Total > 0

# Configure if needed:
sudo sh -c 'echo 2048 > /proc/sys/vm/nr_hugepages'
```

**Check permissions:**
```bash
# Must run with sudo for DPDK access
sudo python3 st2110_test_app.py
```

### Low Frame Rate or Packet Drops

**Tune batch_size in config:**
```yaml
st2110_source:
  batch_size: 4320  # Try adjusting this value
```

Optimal batch_size depends on your video format:
- 1080p60 RGBA: ~5760 packets/frame
- 1080p60 NV12: ~3840 packets/frame
- 4K30 RGBA: ~11520 packets/frame

**Check CPU core allocation:**
```yaml
advanced_network:
  cfg:
    master_core: 3      # Dedicated core for DPDK master
  interfaces:
    - rx:
        queues:
          - cpu_core: 9  # Dedicated core for RX queue
```

## Configuration Details

### Advanced Network Config

The `advanced_network` section configures DPDK:
- **manager**: "dpdk" for DPDK backend
- **memory_regions**: HDS split (headers to CPU, payload to GPU)
- **flows**: Packet steering rules for ST 2110 stream

### ST2110 Source Config

The `st2110_source` section configures the operator:
- **interface_name**: Must match interface name in advanced_network
- **width/height**: Video resolution
- **pixel_format**: Output format (NV12 recommended for NVIDIA)
- **gpu_direct**: Enable zero-copy to GPU
- **header_data_split**: Enable HDS mode

### Holoviz Config

The `visualizer` section configures video display:
- **width/height**: Display window size
- **framerate**: Target display frame rate

## Testing Different Configurations

### Test with RGB Format

```yaml
st2110_source:
  pixel_format: "RGB"
  # RGB uses 3 bytes per pixel
```

### Test with RGBA Format

```yaml
st2110_source:
  pixel_format: "RGBA"
  # RGBA uses 4 bytes per pixel
```

### Test Different Resolutions

```yaml
st2110_source:
  width: 3840
  height: 2160
  framerate: 30
  batch_size: 11520  # Adjust for 4K
```

## Performance Monitoring

The application logs performance metrics:
- Packets received/dropped
- Frame rate
- Bytes received
- Latency (if available)

Example output:
```
[ST2110SourceOp] 1000 packets received, 0 dropped
[ST2110SourceOp] Frame rate: 59.8 FPS
[ST2110SourceOp] Throughput: 2.8 Gbps
```

## Next Steps

After verifying basic reception:
1. Test with different video formats
2. Measure end-to-end latency
3. Add downstream processing operators
4. Test multi-stream reception
5. Optimize for your specific use case

## Related Documentation

- ST2110 Operator: `../../operators/st2110_source/README.md`
- Build Instructions: `../../operators/st2110_source/BUILD.md`
- Development Plan: `../../operators/st2110_source/DEVELOPMENT_PLAN.md`
- Advanced Networking: `../../tutorials/high_performance_networking/README.md`

## Support

For issues:
- Check logs for error messages
- Verify DPDK and network configuration
- Review troubleshooting section above
- File issues at: https://github.com/nvidia-holoscan/holohub/issues
