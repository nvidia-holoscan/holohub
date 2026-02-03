# HoloCat EtherCAT Application Dockerfile

ARG BASE_SDK_VERSION
ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:${BASE_SDK_VERSION}
FROM ${BASE_IMAGE}

# Install system dependencies for EtherCAT and real-time operation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libcap-dev \
    ethtool \
    net-tools \
    iproute2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy EC-Master SDK from build context
# To build: Copy SDK to applications/holocat/ecmaster-sdk/
#   cp -r /path/to/ecmaster applications/holocat/ecmaster-sdk
# Note: Docker COPY doesn't follow symlinks outside build context
RUN mkdir -p /opt/acontis/ecmaster
COPY applications/holocat/ecmaster-sdk /opt/acontis/ecmaster

# Set environment variables for EC-Master SDK
ENV ECMASTER_ROOT=/opt/acontis/ecmaster
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/acontis/ecmaster/lib:/opt/acontis/ecmaster/Bin/Linux/x64

# Create holocat user with necessary privileges
RUN useradd -m -s /bin/bash holocat && \
    usermod -aG sudo holocat
# Create directories for configuration, logs, and scripts
RUN mkdir -p /opt/holocat/config /opt/holocat/logs /opt/holocat/scripts && \
    chown -R holocat:holocat /opt/holocat

# Copy default configuration files and verification script
COPY applications/holocat/configs/holocat_config.yaml /opt/holocat/config/
COPY applications/holocat/configs/eni2.xml /opt/holocat/config/
COPY applications/holocat/scripts/verify_ecmaster.sh /opt/holocat/scripts/
RUN chmod +x /opt/holocat/scripts/verify_ecmaster.sh

# Verify EC-Master installation (required - will fail build if not found)
# Only check critical components during build (headers and libraries)
RUN /opt/holocat/scripts/verify_ecmaster.sh || \
    (echo "Verification failed - checking if SDK is minimally viable..." && \
     test -f "/opt/acontis/ecmaster/SDK/INC/EcMaster.h" && \
     test -f "/opt/acontis/ecmaster/Bin/Linux/x64/libEcMaster.so" && \
     echo "✓ Minimal SDK requirements met for build")

# Set working directory
WORKDIR /workspace

# Switch to holocat user
USER holocat

# Set default environment variables for HoloCat
ENV HOLOCAT_ADAPTER_NAME=eth0
ENV HOLOCAT_ENI_FILE=/opt/holocat/config/eni2.xml
ENV HOLOCAT_CONFIG_DIR=/opt/holocat/config
ENV HOLOCAT_LOG_DIR=/opt/holocat/logs

# Default command
CMD ["/bin/bash"]

# Labels for container metadata
LABEL org.opencontainers.image.title="HoloCat EtherCAT Application"
LABEL org.opencontainers.image.description="Real-time EtherCAT integration with NVIDIA Holoscan SDK"
LABEL org.opencontainers.image.vendor="NVIDIA"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://github.com/nvidia-holoscan/holohub/applications/holocat"