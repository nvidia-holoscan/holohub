#!/usr/bin/env python3

import argparse
import subprocess
import sys
import re
import logging

def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Check system tuning for ANO performance",
        epilog=(
            "Examples:\n"
            "  python check_system.py -cpu-freq       # Check CPU frequency governor\n"
            "  python check_system.py -check-mrrs     # Check MRRS settings for NVIDIA NICs\n\n"
            "  python check_system.py -check-mps      # Check max payload size settings for NVIDIA NICs\n\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["all", "cpu-freq", "check-mrrs", "check-mps", "check-hugepages", "check-gpu-clocks", "check-bar1-size", "check-topo", "check-cmdline"],
        required=True,
        help=(
            "Specify the mode of operation:\n"
            "  all - Perform all checks\n"            
            "  cpu-freq   - Check if the CPU frequency governor is set to 'performance'.\n"
            "  check-mrrs - Check if the Maximum Read Request Size (MRRS) of NVIDIA NICs is set to 4096.\n"
            "  check-mps  - Check if the Maximum Payload Size is set to 256B.\n"
            "  check-hugepages  - Check if hugepages are enabled\n"
            "  check-gpu-clocks - Check GPU clocks\n"
            "  check-bar1-size  - Check the BAR1 size of the GPU\n"
            "  check-topo       - Check the GPU and NIC topology\n"
            "  check-cmdline    - Check the kernel boot parameters\n"
        )
    ) 
    return parser.parse_args()

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
        logging.error("Could not determine online CPUs. File /sys/devices/system/cpu/online not found.")
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
            logging.error(f"CPU {cpu}: Scaling governor file not found. This CPU may not support frequency scaling.")
        except PermissionError:
            logging.error(f"CPU {cpu}: Permission denied while accessing scaling governor file. Run as root.")


def check_mrrs():
    """
    Checks if the Maximum Read Request Size (MRRS) of Mellanox Ethernet controllers
    is set to 4096.
    """
    try:
        # Run lspci to list all PCI devices with detailed information
        result = subprocess.run(
            ["lspci", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Filter lines containing "Ethernet controller" and "Mellanox"
        devices = []
        for line in result.stdout.splitlines():
            if "Ethernet controller" in line and "Mellanox" in line:
                devices.append(line)
        
        if not devices:
            logging.info("No Mellanox Ethernet controllers found on this system.")
            return
        
        for device in devices:
            # Extract the PCI address (e.g., "0000:02:00.0")
            pci_address = device.split()[0]
            
            # Query MRRS for the NIC using setpci
            mrrs_result = subprocess.run(
                ["setpci", "-s", pci_address, "68.w"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Convert MRRS value from hexadecimal to decimal
            mrrs_value = (int(mrrs_result.stdout.strip(), 16) & 0xf000) >> 12
            
            if mrrs_value == 5:
                logging.info(f"{pci_address}: MRRS is correctly set to 4096.")
            else:
                logging.warning(f"{pci_address}: MRRS is set to {2**(7+mrrs_value)}, not 4096.")
    
    except FileNotFoundError:
        logging.error("The required tools (lspci or setpci) are not available on this system.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while checking MRRS: {e}")
        sys.exit(1)    

def check_max_payload_size():
    """
    Checks the Maximum Payload Size (MPS) of Mellanox Ethernet controllers
    from the DevCtl section and ensures it is set to 256 bytes.
    """
    try:
        # Run lspci to list all PCI devices with detailed information
        result = subprocess.run(
            ["lspci", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Filter lines containing "Ethernet controller" and "Mellanox"
        devices = []
        for line in result.stdout.splitlines():
            if "Ethernet controller" in line and "Mellanox" in line:
                devices.append(line)
        
        if not devices:
            logging.info("No Mellanox Ethernet controllers found on this system.")
            return
        
        for device in devices:
            # Extract the PCI address (e.g., "0000:02:00.0")
            pci_address = device.split()[0]
            
            # Query detailed device information using lspci -vv
            mps_result = subprocess.run(
                ["lspci", "-vv", "-s", pci_address],
                capture_output=True,
                text=True,
                check=True
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
                            max_payload_value = int(payload_info.split("MaxPayload")[1].split("bytes")[0].strip())
                            if max_payload_value == 256:
                                logging.info(f"{pci_address}: May Payload Size is correctly set to 256 bytes.")
                            else:
                                logging.warning(f"{pci_address}: May Payload Size is not set to 256 bytes. Found: {max_payload_value} bytes.")
                            break
                    else:
                        logging.error(f"{pci_address}: Unable to find MaxPayload information under DevCtl.")
                    break
            
            if not devctl_found:
                logging.error(f"{pci_address}: DevCtl section not found.")
    
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
        with open('/proc/meminfo', 'r') as file:
            for line in file:
                if 'HugePages_Total' in line:
                    total_hugepages = int(line.split(':')[1].strip())
                elif 'Hugepagesize' in line:
                    hugepage_size_kB = int(line.split(':')[1].strip().split()[0])  # Size in kB

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
    If not, prints the current and maximum values for each GPU.
    """
    try:
        # Run nvidia-smi to get clock information
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "CLOCK"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output of nvidia-smi
        output = result.stdout.splitlines()

        current_gpu = None
        sm_current = None
        sm_max = None
        mem_current = None
        mem_max = None

        for line in output:
            line = line.strip()

            # Detect GPU identifier
            if line.startswith("GPU"):
                current_gpu = line.split()[1].strip(":")
                sm_current = None
                sm_max = None
                mem_current = None
                mem_max = None

            # Parse current clocks under "Clocks" section
            elif sm_current == None and line.startswith("SM") and "MHz" in line:
                sm_current = int(line.split(":")[1].strip().split()[0])
            elif mem_current == None and line.startswith("Memory") and "MHz" in line:
                mem_current = int(line.split(":")[1].strip().split()[0])

            # Parse maximum clocks under "Max Clocks" section
            elif line.startswith("Max Clocks"):
                continue  # Skip the header for Max Clocks section
            elif line.startswith("SM") and "MHz" in line and sm_max is None:
                sm_max = int(line.split(":")[1].strip().split()[0])
            elif line.startswith("Memory") and "MHz" in line and mem_max is None:
                mem_max = int(line.split(":")[1].strip().split()[0])

            # Once all data for a GPU is collected, compare clocks
            if (
                current_gpu is not None
                and sm_current is not None
                and sm_max is not None
                and mem_current is not None
                and mem_max is not None
            ):
                logging.info(f"GPU {current_gpu}: Checking clocks...")

                if sm_current != sm_max:
                    logging.warning(
                        f"GPU {current_gpu}: SM Clock is set to {sm_current} MHz, but should be {sm_max} MHz."
                    )
                else:
                    logging.info(f"GPU {current_gpu}: SM Clock is correctly set to {sm_max} MHz.")

                if mem_current != mem_max:
                    logging.warning(
                        f"GPU {current_gpu}: Memory Clock is set to {mem_current} MHz, but should be {mem_max} MHz."
                    )
                else:
                    logging.info(f"GPU {current_gpu}: Memory Clock is correctly set to {mem_max} MHz.")

                # Reset variables for the next GPU section
                current_gpu = None

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
            ["nvidia-smi", "-q", "-d", "MEMORY"],
            capture_output=True,
            text=True,
            check=True
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
                    logging.warning(f"GPU {current_gpu}: BAR1 size is {bar1_total} MiB. This may indicate an issue.")
                
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
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output of nvidia-smi topo -m
        topo_output = result.stdout.splitlines()

        # Find the header line (contains GPU/NIC labels)
        header_index = 0

        if header_index is None:
            logging.error("Could not find topology table in the nvidia-smi output.")
            return

        # Extract labels (e.g., GPU0, NIC0, etc.)
        labels = topo_output[header_index].split()

        # Parse the topology table rows
        gpu_to_nic_connections = {}
        for row in topo_output[header_index + 1:]:
            row = row.strip()
            if not row:
                continue  # Skip empty lines

            # Split row into columns
            columns = row.split()
            device_label = columns[0]  # First column is the device label (e.g., GPU0)

            # Check connections for GPUs only
            if "GPU" in device_label:
                for col_index, connection_type in enumerate(columns[1:], start=1):
                    target_label = labels[col_index]

                    # Check if it's a GPU-NIC pair
                    if "NIC" in target_label:
                        if device_label not in gpu_to_nic_connections:
                            gpu_to_nic_connections[device_label] = []
                        gpu_to_nic_connections[device_label].append((target_label, connection_type))

        # Verify that each GPU has at least one PIX or PXB connection to a NIC
        for gpu, connections in gpu_to_nic_connections.items():
            has_valid_connection = False
            for nic, connection_type in connections:
                if connection_type in {"PIX", "PXB"}:
                    logging.info(f"GPU {gpu} has at least one PIX/PXB connection to a NIC")
                    has_valid_connection = True
                    break

            if not has_valid_connection:
                for nic, connection_type in connections:
                    logging.error(
                        f"GPU {gpu} does not have a PIX or PXB connection to NIC {nic}. "
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
                logging.warning(f"The kernel command line is missing '{keyword}'. Please ensure it is configured.")
            else:
                logging.info(f"{keyword} found in kernel boot line")

    except FileNotFoundError:
        logging.error("/proc/cmdline not found. Are you sure you're running on Linux?")
    except Exception as e:
        logging.error(f"An unexpected error occurred while checking /proc/cmdline: {e}")        

def main():
    setup_logging()
    args = parse_args()
    if args.mode == "all" or args.mode == "cpu-freq":
        check_cpu_governor()
    if args.mode == "all" or args.mode == "check-mrrs":
        check_mrrs()
    if args.mode == "all" or args.mode == "check-mps":
        check_max_payload_size() 
    if args.mode == "all" or args.mode == "check-hugepages":
        check_hugepages()    
    if args.mode == "all" or args.mode == "check-gpu-clocks":
        check_nvidia_gpu_clocks()
    if args.mode == "all" or args.mode == "check-bar1-size":        
        check_bar1_size()    
    if args.mode == "all" or args.mode == "check-topo":        
        check_topology_connections()         
    if args.mode == "all" or args.mode == "check-cmdline":        
        check_kernel_cmdline()              
          

if __name__ == "__main__":
    main()