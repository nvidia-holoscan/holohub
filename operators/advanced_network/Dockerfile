FROM --platform=$BUILDPLATFORM nvcr.io/nvidia/clara-holoscan/holoscan:v2.0.0-dgpu

ARG UBUNTU_VERSION=22.04
ARG OFED_VERSION=24.01-0.3.3.1
ARG CACHEBUST=1
ARG BUILDARCH
RUN echo "Using build architecture ${BUILDARCH}"

RUN     if [ "${BUILDARCH}" = "arm64" ]; then \
                wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64.tgz && \
                        tar xvf MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64.tgz && \
                        cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64 && \
                        ./mlnxofedinstall --upstream-libs --dpdk --with-mft --upstream-libs --user-space-only --force --without-fw-update && \
                        cd ../ && \
                        rm -fr MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64 && \
                        rm -rf /var/lib/apt/lists/*; \
        else \
                wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-x86_64.tgz && \
                tar xvf MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-x86_64.tgz && \
                cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-x86_64 && \
                ./mlnxofedinstall --upstream-libs --dpdk --with-mft --upstream-libs --user-space-only --force --without-fw-update && \
                cd ../ && \
                rm -fr MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-x86_64 && \
                rm -rf /var/lib/apt/lists/*; \
        fi

RUN apt update && apt install -y python3-pyelftools ninja-build meson 
RUN pip3 install scipy loguru attrs
WORKDIR /opt

RUN    eval `dpkg-architecture` \
        && wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.7.0/host/doca-host_2.7.0-204000-24.04-ubuntu2204_${BUILDARCH}.deb -O doca-host.deb \
        && ls -lh && apt install ./doca-host.deb \
        && apt update \
        && apt install -y doca-all \
        && apt install -y doca-gpu doca-gpu-dev