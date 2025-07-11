#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import re
import subprocess
import sys
from ctypes import CDLL, byref, c_int, create_string_buffer


def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check system tuning for Advanced Network performance",
        epilog=(
            "Examples:\n"
            f"  python {sys.argv[0]} --check cpu-freq    # Check CPU frequency governor\n"
            f"  python {sys.argv[0]} --check mrrs        # Check MRRS settings for NVIDIA NICs\n"
            f"  python {sys.argv[0]} --check mps         # Check max payload size settings\n"
            f"  python {sys.argv[0]} --set mrrs          # Set PCIe MRRS\n\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--check",
        choices=[
            "all",
            "gpudirect",
            "peermem",
            "cpu-freq",
            "mrrs",
            "mps",
            "hugepages",
            "gpu-clocks",
            "bar1-size",
            "topo",
            "cmdline",
            "mtu",
        ],
        help=(
            "Specify the property to check:\n"
            "  all        - Perform all checks\n"
            "  gpudirect  - Check if NVIDIA GPUs support GPUDirect.\n"
            "  peermem    - Check if the nvidia-peermem module is loaded.\n"
            "  cpu-freq   - Check if the CPU frequency governor is set to 'performance'.\n"
            "  mrrs       - Check if the Maximum Read Request Size (MRRS) of NVIDIA NICs is set to 4096.\n"
            "  mps        - Check if the Maximum Payload Size is set to 256B.\n"
            "  hugepages  - Check if hugepages are enabled\n"
            "  gpu-clocks - Check GPU clocks\n"
            "  bar1-size  - Check the BAR1 size of the GPU\n"
            "  topo       - Check the GPU and NIC topology\n"
            "  cmdline    - Check the kernel boot parameters\n"
            "  mtu        - Check MTU of each NVIDIA interface\n"
        ),
    )

    group.add_argument("--set", choices=["mrrs"], help=("  mrrs      - Update MRRS of NICs\n"))

    # Check if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def check_peermem_kernel():
    """
    Check if the nvidia-peermem module for GPUDirect is loaded in the kernel.

    Returns:
        bool: True if nvidia-peermem is loaded, False otherwise
    """
    try:
        # Also check for nvidia_peermem (with underscore)
        result = subprocess.run(
            ["lsmod | grep peermem"], shell=True, capture_output=True, text=True
        )

        if bool(result.stdout.strip()):
            logging.info("nvidia-peermem module is loaded.")
        else:
            logging.warning("nvidia-peermem module is not loaded. GPUDirect may not work.")

    except Exception as e:
        print(f"Error checking for nvidia-peermem module: {e}")
        return False


def check_gpudirect_support():
    """
    Checks if NVIDIA GPUs have access to GPUDirect.
    """
    # Load CUDA Runtime API
    libcuda = CDLL("libcuda.so")

    cudaDevAttrGPUDirectRDMASupported = 116

    result = libcuda.cuInit(0)
    if result != 0:
        logging.error(f"CUDA initialization failed with error code: {result}")
        return
    count = c_int()
    libcuda.cuDeviceGetCount(byref(count))

    for i in range(count.value):
        device = c_int()
        libcuda.cuDeviceGet(byref(device), i)

        name = create_string_buffer(100)
        libcuda.cuDeviceGetName(name, 100, device)

        supported = c_int()
        libcuda.cuDeviceGetAttribute(byref(supported), cudaDevAttrGPUDirectRDMASupported, device)

        if bool(supported.value):
            logging.info(f"GPU {i}: {name.value.decode()} has GPUDirect support.")
        else:
            logging.warning(f"GPU {i}: {name.value.decode()} does not have GPUDirect support.")


def get_nic_info():
    """
    Parses the output of `ibdev2netdev -v` to extract and return a list of tuples,
    where each tuple contains the interface name and its PCIe address.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the IF name and PCIe address
    """
    try:
        # Run ibdev2netdev -v to get detailed information about Mellanox devices
        result = subprocess.run(["ibdev2netdev", "-v"], capture_output=True, text=True, check=True)

        # Parse the output to extract interface names and PCIe addresses
        vals = []
        for line in result.stdout.splitlines():
            match = re.match(r"([\S\:\.]+) .*==>\s+(\S+)", line)
            if match:
                pcie_address = match.group(1)
                interface_name = match.group(2)
                vals.append((interface_name, pcie_address))

        return vals

    except FileNotFoundError:
        print(
            "The ibdev2netdev command is not found. Ensure that it is installed and available in your PATH."
        )
        return [], []
    except subprocess.CalledProcessError as e:
        print(f"Error while executing ibdev2netdev: {e}")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], []


def get_online_cpus():
    """
    Returns a list of online CPUs by reading /sys/devices/system/cpu/online.
    """
    try:
        with open("/sys/devices/system/cpu/online", "r") as f:
            online_cpus = f.read().strip()

        # Parse ranges (e.g., "0-3" -> [0, 1, 2, 3])
        cpu_list = []
        for part in online_cpus.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                cpu_list.extend(range(start, end + 1))
            else:
                cpu_list.append(int(part))

        return cpu_list
    except FileNotFoundError:
        logging.error(
            "Could not determine online CPUs. File /sys/devices/system/cpu/online not found."
        )
        sys.exit(1)


def check_cpu_governor():
    """
    Checks if the CPU frequency governor is set to 'performance' for all online CPUs.
    """
    online_cpus = get_online_cpus()

    for cpu in online_cpus:
        scaling_governor_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"

        try:
            with open(scaling_governor_path, "r") as f:
                governor = f.read().strip()

            if governor == "performance":
                logging.info(f"CPU {cpu}: Governor is correctly set to 'performance'.")
            else:
                logging.warning(f"CPU {cpu}: Governor is set to '{governor}', not 'performance'.")

        except FileNotFoundError:
            logging.error(
                f"CPU {cpu}: Scaling governor file not found. This CPU may not support frequency scaling."
            )
        except PermissionError:
            logging.error(
                f"CPU {cpu}: Permission denied while accessing scaling governor file. Run as root."
            )


def check_mrrs():
    """
    Checks if the Maximum Read Request Size (MRRS) of NVIDIA Ethernet controllers
    is set to 4096.
    """
    try:
        nic_info = get_nic_info()
        for intf in nic_info:
            name = intf[0]
            pci_address = intf[1]

            # Query MRRS for the NIC using setpci
            mrrs_result = subprocess.run(
                ["setpci", "-s", pci_address, "68.w"], capture_output=True, text=True, check=True
            )

            # Convert MRRS value from hexadecimal to decimal
            mrrs_value = (int(mrrs_result.stdout.strip(), 16) & 0xF000) >> 12

            if mrrs_value == 5:
                logging.info(f"{name}/{pci_address}: MRRS is correctly set to 4096.")
            else:
                logging.warning(
                    f"{name}/{pci_address}: MRRS is set to {2**(7+mrrs_value)}, not 4096."
                )

    except FileNotFoundError:
        logging.error("The required tools (lspci or setpci) are not available on this system.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while checking MRRS: {e}")
        sys.exit(1)


def check_max_payload_size():
    """
    Checks the Maximum Payload Size (MPS) of NVIDIA Ethernet controllers
    from the DevCtl section and ensures it is set to 256 bytes.
    """
    try:
        nic_info = get_nic_info()
        for intf in nic_info:
            name = intf[0]
            pci_address = intf[1]

            # Query detailed device information using lspci -vv
            mps_result = subprocess.run(
                ["lspci", "-vv", "-s", pci_address], capture_output=True, text=True, check=True
            )

            # Parse MaxPayload information from the DevCtl section
            lines = mps_result.stdout.splitlines()
            devctl_found = False
            for i, line in enumerate(lines):
                if "DevCtl:" in line:
                    devctl_found = True
                    base_indent = len(line) - len(line.lstrip())  # Indentation level of DevCtl
                    # Look for MaxPayload in subsequent indented lines
                    for j in range(i + 1, len(lines)):
                        current_indent = len(lines[j]) - len(lines[j].lstrip())
                        if current_indent <= base_indent:  # Stop if indentation decreases or ends
                            break
                        if "MaxPayload" in lines[j]:
                            # Extract the actual MaxPayload value from the line
                            payload_info = lines[j].strip()
                            max_payload_value = int(
                                payload_info.split("MaxPayload")[1].split("bytes")[0].strip()
                            )
                            if max_payload_value == 256:
                                logging.info(
                                    f"{name}/{pci_address}: PCIe Max Payload Size is correctly set to 256 bytes."
                                )
                            else:
                                logging.warning(
                                    f"{name}/{pci_address}: PCIe Max Payload Size is not set to 256 bytes. Found: {max_payload_value} bytes."
                                )
                            break
                    else:
                        logging.error(
                            f"{name}/{pci_address}: Unable to find MaxPayload information under DevCtl."
                        )
                    break

            if not devctl_found:
                logging.error(f"{name}/{pci_address}: DevCtl section not found.")

    except FileNotFoundError:
        logging.error("The required tools (lspci) are not available on this system.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while checking Max Payload Size: {e}")
        sys.exit(1)


def check_hugepages():
    """
    Checks if hugepages are allocated and ensures that the total allocated hugepage memory
    is at least 500 MB.
    """
    try:
        # Initialize variables
        total_hugepages = 0
        hugepage_size_kB = 0

        # Read /proc/meminfo for hugepage details
        with open("/proc/meminfo", "r") as file:
            for line in file:
                if "HugePages_Total" in line:
                    total_hugepages = int(line.split(":")[1].strip())
                elif "Hugepagesize" in line:
                    hugepage_size_kB = int(line.split(":")[1].strip().split()[0])  # Size in kB

        # Check if hugepages are allocated
        if total_hugepages > 0:
            # Calculate total memory allocated to hugepages in MB
            hugepage_size_MB = hugepage_size_kB / 1024  # Convert kB to MB
            total_allocated_memory_MB = total_hugepages * hugepage_size_MB

            logging.info(f"HugePages_Total: {total_hugepages}")
            logging.info(f"HugePage Size: {hugepage_size_MB:.2f} MB")
            logging.info(f"Total Allocated HugePage Memory: {total_allocated_memory_MB:.2f} MB")

            # Check if the total memory meets the 500 MB requirement
            if total_allocated_memory_MB >= 500:
                logging.info("Hugepages are sufficiently allocated with at least 500 MB.")
                return True
            else:
                logging.warning("Hugepages are allocated but do not meet the 500 MB requirement.")
                return False
        else:
            logging.warning("No hugepages are allocated.")
            return False

    except FileNotFoundError:
        logging.error("/proc/meminfo not found. Are you sure you're running on Linux?")
        return False
    except Exception as e:
        logging.error(f"An error occurred while checking hugepages: {e}")
        return False


def check_nvidia_gpu_clocks():
    """
    Checks all NVIDIA GPUs to ensure that the SM clock and memory clock are set to their maximum values.
    If not, logs the current and maximum values for each GPU.
    """
    try:
        # Define the fields to query
        fields = ["clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem"]
        query = ",".join(fields)

        # Run nvidia-smi to get clock information
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output of nvidia-smi
        output = result.stdout.strip().splitlines()

        for idx, line in enumerate(output):
            sm_current, sm_max, mem_current, mem_max = map(int, line.split(","))

            logging.debug(f"GPU {idx}: Checking clocks...")

            # Some GPUs have a boost clock that appears as the "max clock", but when you set the
            # GPU to that frequency it will not report as that with nvidia-smi. For example:
            # nvidia-smi --lock-gpu-clocks 3105 --mode 1
            # GPU clocks set to "(gpuClkMin 3105, gpuClkMax 3105)" for GPU 00000005:09:00.0
            #
            # Clocks
            #  SM                                : 2730 MHz
            #
            # Anecdotally if a user has not set their clocks at all the value will be very low,
            # around 100-300MHz. Having a check within 500MHz should be sufficient to catch this.
            sm_margin = 500
            if abs(sm_current - sm_max) > sm_margin:
                logging.warning(
                    f"GPU {idx}: SM Clock is set to {sm_current} MHz, but should be within {sm_margin} MHz of the {sm_max} MHz theoretical Max."
                )
            elif sm_current < sm_max:
                logging.info(
                    f"GPU {idx}: SM Clock is correctly set to {sm_current} MHz (within {sm_margin} of the {sm_max} MHz theoretical Max)."
                )
            else:
                logging.info(f"GPU {idx}: SM Clock is correctly set to {sm_current} MHz.")

            # nvidia-smi has a bug where the memory clock is reported as 1 MHz less than the max in
            # some cases
            if abs(mem_current - mem_max) > 1:
                logging.warning(
                    f"GPU {idx}: Memory Clock is set to {mem_current} MHz, but should be {mem_max} MHz."
                )
            else:
                logging.info(f"GPU {idx}: Memory Clock is correctly set to {mem_current} MHz.")

    except FileNotFoundError:
        logging.error("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while querying NVIDIA GPUs: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def check_bar1_size():
    """
    Checks the BAR1 size of all NVIDIA GPUs using nvidia-smi.
    Logs the BAR1 size for each GPU and ensures it is non-zero.
    """
    try:
        # Run nvidia-smi to get BAR1 memory information
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "MEMORY"], capture_output=True, text=True, check=True
        )

        # Parse the output of nvidia-smi
        output = result.stdout.splitlines()

        current_gpu = None
        bar1_total = None

        for i, line in enumerate(output):
            line = line.strip()

            # Detect GPU identifier
            if line.startswith("GPU"):
                current_gpu = line.split()[1].strip(":")
                bar1_total = None

            # Parse BAR1 total size under "BAR1 Memory Usage" section
            elif "BAR1 Memory Usage" in line:
                continue  # Skip the header for BAR1 Memory Usage section
            elif "Total" in line and "BAR1" in output[i - 1]:
                bar1_total_str = line.split(":")[1].strip()
                if "MiB" in bar1_total_str:
                    bar1_total = int(bar1_total_str.split()[0])

            # Once BAR1 size is found, log it
            if current_gpu is not None and bar1_total is not None:
                if bar1_total > 1024:
                    logging.info(f"GPU {current_gpu}: BAR1 size is {bar1_total} MiB.")
                else:
                    logging.warning(
                        f"GPU {current_gpu}: BAR1 size is {bar1_total} MiB. This may indicate an issue."
                    )

                # Reset variables for the next GPU section
                current_gpu = None

    except FileNotFoundError:
        logging.error("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while querying NVIDIA GPUs: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def check_topology_connections():
    """
    Executes `nvidia-smi topo -m`, parses its output, and ensures that every GPU has at least one PIX or PXB connection to a NIC.
    If not, logs an error specifying the GPU, NIC, and the actual connection type.
    """
    try:
        # Run nvidia-smi topo -m to get topology information
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True
        )

        # Parse the output of nvidia-smi topo -m
        topo_output = result.stdout.splitlines()

        # Find the header line (contains GPU/NIC labels)
        header_index = 0

        if header_index is None:
            logging.error("Could not find topology table in the nvidia-smi output.")
            return

        # Extract labels (e.g., GPU0, NIC0, etc.)
        header = topo_output[header_index].strip()
        labels = []
        for label in header.split():
            if label.startswith("GPU") or label.startswith("NIC"):
                labels.append(label)

        # Parse the topology table rows
        gpu_to_nic_connections = {}
        for row_idx, row in enumerate(topo_output[header_index + 1 :]):
            row = row.strip()
            if not row:
                continue  # Skip empty lines

            # Split row into columns
            columns = row.split()
            device_label = columns[0]  # First column is the device label (e.g., GPU0)

            # Check connections for GPUs only
            if "GPU" in device_label:
                # We need to align the columns with the labels
                # The first column after the device label corresponds to the first label
                for label_idx, label in enumerate(labels):
                    if "NIC" in label:
                        # label_idx + 1 because columns[0] is the device label
                        connection_type = columns[label_idx + 1]

                        if device_label not in gpu_to_nic_connections:
                            gpu_to_nic_connections[device_label] = []
                        gpu_to_nic_connections[device_label].append((label, connection_type))

        # Verify that each GPU has at least one PIX or PXB connection to a NIC
        for gpu, connections in gpu_to_nic_connections.items():
            has_valid_connection = False
            for nic, connection_type in connections:
                if connection_type in {"PIX", "PXB"}:
                    logging.info(f"{gpu} has a {connection_type} connection to {nic}")
                    has_valid_connection = True
                    break

            if not has_valid_connection:
                for nic, connection_type in connections:
                    logging.warning(
                        f"{gpu} does not have a PIX or PXB connection to {nic}. "
                        f"Current connection type: {connection_type}."
                    )

    except FileNotFoundError:
        logging.error("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while executing nvidia-smi topo -m: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def check_kernel_cmdline():
    """
    Checks if the words "isolcpus", "rcu_nocbs", and "irqaffinity" appear in /proc/cmdline.
    Logs a separate warning for each word that is missing.
    """
    try:
        # Read the contents of /proc/cmdline
        with open("/proc/cmdline", "r") as file:
            cmdline = file.read().strip()

        # Check for each required word
        required_keywords = ["isolcpus", "rcu_nocbs", "irqaffinity"]
        for keyword in required_keywords:
            if keyword not in cmdline:
                logging.warning(
                    f"The kernel command line is missing '{keyword}'. Please ensure it is configured."
                )
            else:
                logging.info(f"{keyword} found in kernel boot line")

    except FileNotFoundError:
        logging.error("/proc/cmdline not found. Are you sure you're running on Linux?")
    except Exception as e:
        logging.error(f"An unexpected error occurred while checking /proc/cmdline: {e}")


def check_mtu_size():
    """
    Checks the MTU size of each NVIDIA NIC using the sysfs interface and prints a warning if it's not over 1500 bytes.
    """
    try:
        nic_info = get_nic_info()
        for intf in nic_info:
            iface = intf[0]

            # Check MTU size for each NVIDIA NIC using sysfs
            mtu_path = f"/sys/class/net/{iface}/mtu"
            if os.path.exists(mtu_path):
                with open(mtu_path, "r") as f:
                    mtu_value = int(f.read().strip())
                    if mtu_value <= 1518:
                        logging.warning(
                            f"Interface {iface} has an MTU of {mtu_value} bytes. "
                            "If possible use larger frame sizes ( > 1518B) for better performance"
                        )
                    else:
                        logging.info(
                            f"Interface {iface} has an acceptable MTU of {mtu_value} bytes."
                        )
            else:
                logging.error(f"MTU file for interface {iface} does not exist.")

    except FileNotFoundError:
        logging.error(
            "The ibdev2netdev command is not found. Ensure that it is installed and available in your PATH."
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while executing a command: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def update_mrrs_for_nvidia_devices():
    """
    Updates the PCIe Maximum Read Request Size (MRRS) to 4096 for all Mellanox devices,
    preserving the lower 12 bits of the current setting.
    """
    try:
        nic_info = get_nic_info()
        for intf in nic_info:
            pci_address = intf[1]

            try:
                # Read the current MRRS value
                read_result = subprocess.run(
                    ["setpci", "-s", pci_address, "68.w"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                current_value_hex = read_result.stdout.strip()
                current_value = int(current_value_hex, 16)

                # Calculate new value: keep lower 12 bits, set upper 4 bits to 5 (for 4096 bytes)
                new_value = (current_value & 0x0FFF) | (0x5 << 12)

                # Write the new MRRS value back
                subprocess.run(["setpci", "-s", pci_address, f"68.w={new_value:04x}"], check=True)
                logging.info(
                    f"Successfully updated MRRS to 4096 for device at PCIe address {pci_address}={hex(new_value)}."
                )
            except subprocess.CalledProcessError as e:
                logging.error(
                    f"Failed to update MRRS for device at PCIe address {pci_address}: {e}"
                )

    except FileNotFoundError:
        logging.error(
            "The ibdev2netdev or setpci command is not found. Ensure that they are installed and available in your PATH."
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while executing a command: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def main():
    setup_logging()
    args = parse_args()

    if args.check is not None:
        if args.check == "all" or args.check == "cpu-freq":
            check_cpu_governor()
        if args.check == "all" or args.check == "mrrs":
            check_mrrs()
        if args.check == "all" or args.check == "mps":
            check_max_payload_size()
        if args.check == "all" or args.check == "hugepages":
            check_hugepages()
        if args.check == "all" or args.check == "gpu-clocks":
            check_nvidia_gpu_clocks()
        if args.check == "all" or args.check == "bar1-size":
            check_bar1_size()
        if args.check == "all" or args.check == "topo":
            check_topology_connections()
        if args.check == "all" or args.check == "cmdline":
            check_kernel_cmdline()
        if args.check == "all" or args.check == "mtu":
            check_mtu_size()
        if args.check == "all" or args.check == "gpudirect":
            check_gpudirect_support()
        if args.check == "all" or args.check == "peermem":
            check_peermem_kernel()
    elif args.set is not None:
        if args.set == "mrrs":
            update_mrrs_for_nvidia_devices()


if __name__ == "__main__":
    if os.geteuid() != 0:
        sys.exit("This script must be run as root! Please use 'sudo' to execute it.")

    main()
