FROM --platform=$BUILDPLATFORM nvcr.io/nvidia/clara-holoscan/holoscan:v2.0.0-dgpu

ARG UBUNTU_VERSION=22.04
ARG OFED_VERSION=24.01-0.3.3.1
ARG CACHEBUST=1
ARG BUILDARCH
RUN echo "Using build architecture ${BUILDARCH}"
RUN echo "Using architecture $TARGETPLATFORM"

RUN apt update && apt install -y python3-pyelftools ninja-build meson 
RUN pip3 install scipy loguru attrs
WORKDIR /opt

RUN    eval `dpkg-architecture` \
        && wget https://www.mellanox.com/downloads/DOCA/DOCA_v2.7.0/host/doca-host_2.7.0-204000-24.04-ubuntu2204_${BUILDARCH}.deb -O doca-host.deb \
        && ls -lh && apt install ./doca-host.deb \
        && apt update \
        && apt install -y doca-all \
        && apt install -y doca-gpu doca-gpu-dev
        