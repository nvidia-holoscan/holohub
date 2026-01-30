#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# HoloCat EC-Master SDK Verification Script

set -e

echo "============================================="
echo "HoloCat EC-Master SDK Verification"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}❌${NC} $message"
    fi
}

# Check for ECMASTER_ROOT environment variable
echo "1. Checking ECMASTER_ROOT environment variable..."
if [ -z "$ECMASTER_ROOT" ]; then
    print_status "ERROR" "ECMASTER_ROOT environment variable not set"
    echo "   Set it to your EC-Master installation path:"
    echo "   export ECMASTER_ROOT=/path/to/ecm"
    exit 1
else
    print_status "OK" "ECMASTER_ROOT set to: $ECMASTER_ROOT"
fi

# Check if ECMASTER_ROOT directory exists
echo ""
echo "2. Checking EC-Master SDK directory structure..."
if [ ! -d "$ECMASTER_ROOT" ]; then
    print_status "ERROR" "EC-Master root directory not found: $ECMASTER_ROOT"
    exit 1
else
    print_status "OK" "EC-Master root directory exists"
fi

# Check SDK include directory
if [ ! -d "$ECMASTER_ROOT/SDK/INC" ]; then
    print_status "ERROR" "EC-Master SDK include directory not found: $ECMASTER_ROOT/SDK/INC"
    exit 1
else
    print_status "OK" "SDK include directory found"
fi

# Check for main header file
if [ ! -f "$ECMASTER_ROOT/SDK/INC/EcMaster.h" ]; then
    print_status "ERROR" "EcMaster.h not found in: $ECMASTER_ROOT/SDK/INC/"
    exit 1
else
    print_status "OK" "EcMaster.h header file found"
fi

# Check for Linux-specific headers
if [ ! -f "$ECMASTER_ROOT/SDK/INC/Linux/EcOsPlatform.h" ]; then
    print_status "WARN" "Linux platform headers not found (may be optional)"
else
    print_status "OK" "Linux platform headers found"
fi

# Determine architecture
ARCH="x64"
if [ "$(uname -m)" = "aarch64" ]; then
    ARCH="arm64"
fi

# Check for libraries
echo ""
echo "3. Checking EC-Master libraries (architecture: $ARCH)..."
LIB_DIR="$ECMASTER_ROOT/Bin/Linux/$ARCH"
if [ ! -d "$LIB_DIR" ]; then
    print_status "ERROR" "Library directory not found: $LIB_DIR"
    exit 1
else
    print_status "OK" "Library directory found"
fi

# Check main library
if [ ! -f "$LIB_DIR/libEcMaster.so" ]; then
    print_status "ERROR" "libEcMaster.so not found in: $LIB_DIR"
    exit 1
else
    print_status "OK" "libEcMaster.so found"
    # Get library info
    LIB_SIZE=$(ls -lh "$LIB_DIR/libEcMaster.so" | awk '{print $5}')
    echo "   Library size: $LIB_SIZE"
fi

# Check for link layer libraries
echo ""
echo "4. Checking EtherCAT link layer libraries..."
LINK_LIBS=("libemllSockRaw.so" "libemllDpdk.so" "libemllIntelGbe.so" "libemllRTL8169.so" "libemllVlan.so")
FOUND_LIBS=0

for lib in "${LINK_LIBS[@]}"; do
    if [ -f "$LIB_DIR/$lib" ]; then
        print_status "OK" "$lib found"
        FOUND_LIBS=$((FOUND_LIBS + 1))
    else
        print_status "WARN" "$lib not found (may be optional)"
    fi
done

if [ $FOUND_LIBS -eq 0 ]; then
    print_status "ERROR" "No link layer libraries found"
    exit 1
else
    print_status "OK" "$FOUND_LIBS link layer libraries available"
fi

# Check version information
echo ""
echo "5. Checking EC-Master version..."
if [ -f "$ECMASTER_ROOT/EcVersion.txt" ]; then
    VERSION=$(cat "$ECMASTER_ROOT/EcVersion.txt" 2>/dev/null || echo "Unknown")
    print_status "OK" "Version file found: $VERSION"
elif [ -f "$ECMASTER_ROOT/SDK/INC/EcVersion.h" ]; then
    # Extract version from header file
    MAJ=$(grep "#define EC_VERSION_MAJ" "$ECMASTER_ROOT/SDK/INC/EcVersion.h" | awk '{print $3}')
    MIN=$(grep "#define EC_VERSION_MIN" "$ECMASTER_ROOT/SDK/INC/EcVersion.h" | awk '{print $3}')
    SP=$(grep "#define EC_VERSION_SERVICEPACK" "$ECMASTER_ROOT/SDK/INC/EcVersion.h" | awk '{print $3}')
    BUILD=$(grep "#define EC_VERSION_BUILD" "$ECMASTER_ROOT/SDK/INC/EcVersion.h" | awk '{print $3}')
    if [ -n "$MAJ" ] && [ -n "$MIN" ] && [ -n "$SP" ] && [ -n "$BUILD" ]; then
        VERSION="$MAJ.$MIN.$SP.$BUILD"
        print_status "OK" "Version extracted from header: $VERSION"
    else
        print_status "WARN" "Could not determine version"
    fi
else
    print_status "WARN" "Version information not found"
fi

# Check system requirements
echo ""
echo "6. Checking system requirements..."

# Check for required capabilities
if command -v getcap >/dev/null 2>&1; then
    print_status "OK" "getcap utility available"
else
    print_status "WARN" "getcap utility not found (install libcap2-bin)"
fi

# Check for real-time kernel
if uname -r | grep -q rt; then
    print_status "OK" "Real-time kernel detected: $(uname -r)"
else
    print_status "WARN" "Standard kernel detected: $(uname -r)"
    echo "   For best performance, consider using a real-time kernel (PREEMPT_RT)"
fi

# Check for network interfaces
echo ""
echo "7. Checking network interfaces..."
if command -v ip >/dev/null 2>&1; then
    INTERFACES=$(ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | wc -l)
    if [ $INTERFACES -gt 0 ]; then
        print_status "OK" "$INTERFACES network interfaces available"
        echo "   Available interfaces:"
        ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | awk '{print "   - " $2}' | sed 's/:$//'
    else
        print_status "WARN" "No network interfaces found (excluding loopback)"
    fi
else
    print_status "WARN" "ip command not found"
fi

# Check build tools
echo ""
echo "8. Checking build requirements..."
if command -v cmake >/dev/null 2>&1; then
    CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
    print_status "OK" "CMake found: $CMAKE_VERSION"
else
    print_status "ERROR" "CMake not found (required for building)"
fi

if command -v gcc >/dev/null 2>&1; then
    GCC_VERSION=$(gcc --version | head -n1 | awk '{print $4}')
    print_status "OK" "GCC found: $GCC_VERSION"
else
    print_status "WARN" "GCC not found"
fi

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,$//')
    print_status "OK" "CUDA found: $CUDA_VERSION"
else
    print_status "WARN" "CUDA nvcc not found"
fi

# Summary
echo ""
echo "============================================="
echo "Verification Summary"
echo "============================================="
print_status "OK" "EC-Master SDK verification completed successfully"
echo ""
echo "Next steps:"
echo "1. Build HoloCat: ./holohub build holocat"
echo "2. Configure network adapter in holocat_config.yaml"
echo "3. Create or obtain EtherCAT ENI configuration file"
echo "4. Run HoloCat: ./holohub run holocat"
echo ""
echo "For more information, see: applications/holocat/README.md"
