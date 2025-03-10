#!/bin/bash

# Set the maximum send and receive socket buffer sizes to match what
# is used in the UDPv4 section of the QoS profile (qos_profiles.xml).
# For more details, see the RTI Connext guide to Improve DDS Network
# Performance on Linux Systems:
#   https://community.rti.com/howto/improve-rti-connext-dds-network-performance-linux-systems
sudo sysctl -w net.core.rmem_max="4194304"
sudo sysctl -w net.core.wmem_max="4194304"
