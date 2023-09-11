FROM --platform=$BUILDPLATFORM nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu

ARG OFED_VERSION=5.8-3.0.7.0
ARG UBUNTU_VERSION=20.04
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

ARG DPDK_VERSION=22.11.1
RUN apt update && apt install -y python3-pyelftools ninja-build meson 

ADD https://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz /tmp/
RUN cd /tmp && tar xf dpdk-${DPDK_VERSION}.tar.xz
COPY ./dpdk_patches/*.patch /tmp/dpdk-stable-${DPDK_VERSION}
WORKDIR /tmp/dpdk-stable-${DPDK_VERSION}/
RUN patch --ignore-whitespace --fuzz 3  config/arm/meson.build /tmp/dpdk-stable-${DPDK_VERSION}/dpdk.nvidia.patch 
RUN patch --ignore-whitespace --fuzz 3  drivers/gpu/cuda/devices.h /tmp/dpdk-stable-${DPDK_VERSION}/devices.h.patch
RUN patch --ignore-whitespace --fuzz 3  drivers/gpu/cuda/cuda.c /tmp/dpdk-stable-${DPDK_VERSION}/cuda.c.patch
RUN CFLAGS=-I/usr/local/cuda/include meson build -Dplatform=generic -Dc_args=-I/usr/local/cuda/include \
          -Ddisabled_drivers=baseband/*,bus/ifpga/*,common/cpt,common/dpaax,common/iavf,common/octeontx,common/octeontx2,crypto/nitrox,net/ark,net/atlantic,net/avp,net/axgbe,net/bnx2x,net/bnxt,net/cxgbe,net/e1000,net/ena,net/enic,net/fm10k,net/hinic,net/hns3,net/i40e,net/ixgbe,vdpa/ifc,net/igc,net/liquidio,net/netvsc,net/nfp,net/qede,net/sfc,net/thunderx,net/vdev_netvsc,net/vmxnet3,regex/octeontx2,
RUN ninja -C build install
RUN rm /tmp/dpdk-${DPDK_VERSION}.tar.xz

RUN pip3 install scipy loguru attrs

