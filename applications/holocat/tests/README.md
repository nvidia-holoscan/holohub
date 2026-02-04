# HoloCat Tests

Tests for the HoloCat EtherCAT application.

## Quick Start

```bash
# Build and run all tests
./holohub build holocat --local
cd build/holocat
ctest --output-on-failure

# For hardware tests (optional, requires EtherCAT hardware):
export ECMASTER_ROOT=/path/to/ec-master
export HOLOCAT_TEST_ADAPTER=eth0
export HOLOCAT_TEST_ENI=/path/to/eni.xml
sudo setcap 'cap_net_raw=ep' applications/holocat/tests/test_integration_hardware
ctest -R holocat_hardware --output-on-failure
```

## Test Structure

```
tests/
├── test_hc_data_tx_op.cpp         # TX operator unit tests
├── test_hc_data_rx_op.cpp         # RX operator unit tests
├── test_holocat_op.cpp            # Core operator unit tests
├── test_integration_mocked.cpp    # Integration test with mocked hardware
├── test_integration_hardware.cpp  # Hardware in-the-loop tests (optional)
└── mocks/
    ├── mock_ec_master.hpp         # Mock EC-Master SDK
    └── test_helpers.hpp           # Test utilities
```

## Running Tests

```bash
# All tests
ctest --test-dir build/holocat -R holocat

# Unit tests only (no hardware)
ctest --test-dir build/holocat -R holocat_unit

# Mocked integration tests
ctest --test-dir build/holocat -R holocat_mocked

# Hardware tests (requires setup)
ctest --test-dir build/holocat -R holocat_hardware
```

## Hardware Tests Setup

Hardware tests are **automatically skipped** if hardware is unavailable.

### Prerequisites

```bash
export ECMASTER_ROOT=/opt/ec-master
export HOLOCAT_TEST_ADAPTER=eth0
export HOLOCAT_TEST_ENI=/tmp/test_eni.xml

# Grant network access (EtherCAT requires raw sockets)
sudo setcap 'cap_net_raw=ep' build/holocat/applications/holocat/tests/test_integration_hardware
```

**Requirements:**
- EtherCAT network adapter
- At least one connected slave device
- Valid ENI configuration file

## Troubleshooting

### Permission Errors

```bash
# Grant capability (recommended)
sudo setcap 'cap_net_raw=ep' build/holocat/applications/holocat/tests/test_integration_hardware

# Or run with sudo
sudo -E ./applications/holocat/tests/test_integration_hardware
```

**Note:** Reapply `setcap` after rebuilding.

### EC-Master Not Found

```bash
export ECMASTER_ROOT=/path/to/ec-master-sdk
./holohub build holocat --local --reconfigure
```

## References

- [HoloCat README](../README.md)
- [Google Test Documentation](https://google.github.io/googletest/)
- [Holoscan SDK Documentation](https://docs.nvidia.com/holoscan/)
