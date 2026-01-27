# ST2110 Test Application Setup

## Quick Setup (Python Virtual Environment)

This application uses a Python virtual environment with Holoscan installed via pip.

### 1. Create Virtual Environment

```bash
cd applications/st2110_test
python3 -m venv venv
```

### 2. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install Holoscan for CUDA 13
pip install holoscan-cu13

# Verify installation
python -c "import holoscan; print('Holoscan version:', holoscan.__version__)"
# Should output: Holoscan version: 3.8.0 (or later)
```

**Note**: The venv is already created and holoscan-cu13 is installed if you cloned this directory after initial setup.

### 3. Configure Network

Edit `st2110_test_config.yaml` and update your NIC's PCI address:

```bash
# Find your NIC's PCI address
lspci | grep -i ethernet
# Example output: 0005:03:00.0 Ethernet controller: NVIDIA Corporation

# Update in config:
nano st2110_test_config.yaml
# Change: address: "0005:03:00.0"  # ← Your PCI address
```

### 4. Bind NIC to DPDK

```bash
# Take interface down
sudo ifconfig mgbe0_0 down

# Bind to DPDK (requires vfio-pci driver)
sudo dpdk-devbind.py --bind=vfio-pci 0005:03:00.0

# Verify binding
sudo dpdk-devbind.py --status
```

### 5. Run Application

**Option A: Using the run script (recommended)**
```bash
./run.sh
# The script automatically handles sudo and venv activation
```

**Option B: Manual run**
```bash
sudo ./venv/bin/python st2110_test_app.py
```

**Note**: Must run with sudo for DPDK access.

## Dependencies Installed

When you run `pip install holoscan-cu13`, it installs:
- **holoscan-cu13** (3.8.0): Main Holoscan SDK
- **cupy-cuda13x** (13.6.0): CUDA arrays for Python
- **numpy** (2.3.4): Numerical computing
- **pillow** (12.0.0): Image processing
- **cloudpickle** (3.1.2): Serialization
- **wheel-axle-runtime** (0.0.7): Runtime support

## Verifying Installation

Test all imports:
```bash
./venv/bin/python -c "
from holoscan.core import Application
from holoscan.operators import HolovizOp
import cupy as cp
print('✓ All imports successful')
print('✓ Holoscan:', __import__('holoscan').__version__)
print('✓ CuPy:', cp.__version__)
print('✓ CUDA available:', cp.cuda.is_available())
"
```

Expected output:
```
✓ All imports successful
✓ Holoscan: 3.8.0
✓ CuPy: 13.6.0
✓ CUDA available: True
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'holoscan'"

**Solution**: Activate the virtual environment first
```bash
source venv/bin/activate
python st2110_test_app.py
```

Or use the run script:
```bash
./run.sh
```

### "CUDA not available" or CuPy errors

**Solution**: Verify CUDA 13.0 is installed
```bash
/usr/local/cuda/bin/nvcc --version
# Should show: release 13.0
```

If CUDA is in a different location:
```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### DPDK initialization fails

**Solution**: Check huge pages and permissions
```bash
# Check huge pages
cat /proc/meminfo | grep Huge

# Configure if needed
sudo sh -c 'echo 2048 > /proc/sys/vm/nr_hugepages'

# Verify vfio-pci driver is loaded
lsmod | grep vfio
```

### NIC binding fails

**Solution**: Ensure kernel modules are loaded
```bash
# Load vfio-pci
sudo modprobe vfio-pci

# Check if NIC is already bound
sudo dpdk-devbind.py --status

# If stuck, unbind first
sudo dpdk-devbind.py --unbind 0005:03:00.0
sudo dpdk-devbind.py --bind=vfio-pci 0005:03:00.0
```

## Updating Holoscan

To update to a newer version:
```bash
source venv/bin/activate
pip install --upgrade holoscan-cu13
```

## Rebuilding Virtual Environment

If you need to start fresh:
```bash
# Remove old venv
rm -rf venv

# Create new one
python3 -m venv venv
source venv/bin/activate
pip install holoscan-cu13
```

## Alternative: System-Wide Installation

If you prefer system-wide installation instead of venv:
```bash
pip3 install --user holoscan-cu13
python3 st2110_test_app.py
```

## Next Steps

After successful setup:
1. Verify ST 2110 video source is streaming
2. Run the application: `./run.sh`
3. Check console for packet reception logs
4. Video window should display the stream

See `README.md` for full documentation and usage instructions.
