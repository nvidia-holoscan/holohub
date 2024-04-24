FROM --platform=$BUILDPLATFORM nvcr.io/nvidia/clara-holoscan/holoscan:v1.0.3-dgpu

ARG TARGETPLATFORM
ARG UBUNTU_VERSION=2204
ARG DOCA_VERSION=2.6.0
ARG DOCA_BUILD=0.0.1.2.6.0058.1.24.01.0.3.3.1
ARG DOCA_PACKAGES="doca-networking doca-ofed doca-gpu doca-gpu-dev openvswitch-switch=2.17.8-1  openvswitch-common=2.17.8-1"

RUN echo "Using architecture $TARGETPLATFORM"

RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
             wget https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host-repo-ubuntu${UBUNTU_VERSION}_${DOCA_VERSION}-${DOCA_BUILD}_amd64.deb && \
                dpkg -i doca-host-repo-ubuntu${UBUNTU_VERSION}_${DOCA_VERSION}-${DOCA_BUILD}_amd64.deb && \
                apt update && \
                        apt install -y ${DOCA_PACKAGES}; \
    else \
             wget https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host-repo-ubuntu${UBUNTU_VERSION}_${DOCA_VERSION}-${DOCA_BUILD}_arm64.deb && \
                dpkg -i doca-host-repo-ubuntu${UBUNTU_VERSION}_${DOCA_VERSION}-${DOCA_BUILD}_arm64.deb && \
                apt update && \
                        apt install -y ${DOCA_PACKAGES}; \        
    fi

RUN pip3 install scipy numpy

