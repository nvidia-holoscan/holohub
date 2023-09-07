ARG VULKAN_SDK_VERSION=1.3.216.0

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu as base

############################################################
# Vulkan SDK
#
# Use the SDK because we need the newer Vulkan headers and the newer shader compiler than provided
# by the Ubuntu deb packages. These are compile time dependencies, we still use the Vulkan loaded
# and the Vulkan validation layer as runtime components provided by Ubuntu packages because that's
# what the user will have on their installations.
############################################################
FROM base as vulkansdk-builder
ARG VULKAN_SDK_VERSION

WORKDIR /opt/vulkansdk

# Note there is no aarch64 binary version to download, therefore for aarch64 we also download the x86_64 version which
# includes the source. Then remove the binaries and build the aarch64 version from source.
RUN wget -nv --show-progress --progress=bar:force:noscroll \
    https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
RUN if [ $(uname -m) == "aarch64" ]; then \
    cd ${VULKAN_SDK_VERSION} \
    && rm -rf x86_64 \
    && ./vulkansdk shaderc glslang headers; \
    fi

FROM base as dev

ARG DEBIAN_FRONTEND=noninteractive

############################################################
# VSCode devcontainer-compatible Dockerfile
# (https://code.visualstudio.com/docs/remote/create-dev-container)
############################################################

# Options for setup script
ARG INSTALL_ZSH="true"
# don't upgrade packages, some packages are pinned by the base image
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=holoscan-openxr
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY .devcontainer/library-scripts/common-debian.sh /tmp/library-scripts/
RUN yes | unminimize 2>&1 \
    && bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && rm -rf /tmp/library-scripts

# Install Monado OpenXR runtime. See
# https://monado.freedesktop.org/getting-started.html#monado-installation
RUN apt update \
    && apt install --no-install-recommends -y software-properties-common \
    && add-apt-repository ppa:monado-xr/monado \
    && apt update \
    && apt install --no-install-recommends -y \
        libopenxr-loader1 libopenxr-dev libopenxr1-monado \
    && rm -rf /var/lib/apt/lists/*

# Install Magic Leap OpenXR runtime and set it as the default active runtime
COPY applications/volume_rendering_xr/thirdparty/magicleap/ /tmp/magicleap
RUN mv /tmp/magicleap/windrunner-$(uname -m) /opt/windrunner \
    && update-alternatives --install /etc/xdg/openxr/1/active_runtime.json \
        openxr1-active-runtime \
        /opt/windrunner/lib/windrunner/openxr_windrunner.json 60 \
    && update-alternatives --set openxr1-active-runtime \
        /opt/windrunner/lib/windrunner/openxr_windrunner.json \
    && rm -rf /tmp/magicleap
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/windrunner/lib"
ENV LR_TRANSPORT_CERT_PATH=/opt/windrunner/share/windrunner/server.crt
ENV LR_TRANSPORT_PKEY_PATH=/opt/windrunner/share/windrunner/server.key
ENV WINDRUNNER_ENABLE_AUDIO=0
ENV PATH="${PATH}:/opt/windrunner/bin"

# OpenXR dependencies
RUN apt update \
    && apt install --no-install-recommends -y \
        build-essential \
        libevent-core-2.1-7 \
        libgl1-mesa-dev \
        libx11-xcb-dev \
        libxcb-dri2-0-dev \
        libxcb-glx0-dev \
        libxcb-icccm4-dev \
        libxcb-keysyms1-dev \
        libxcb-randr0-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        mesa-common-dev \
    && rm -rf /var/lib/apt/lists/*

# Additional tooling
RUN apt update \
    && apt install --no-install-recommends -y \
        gdb \
        git-lfs \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_GENERATOR=Ninja

# Fix build error by downgrading markupsafe
#  [build] ImportError: cannot import name 'soft_unicode' from 'markupsafe' (/usr/local/lib/python3.8/dist-packages/markupsafe/__init__.py)
#  [build] [15/88   7% :: 0.058] Generating /workspace/holoscan-openxr/build/_deps/openxr-build/include/openxr/openxr_reflection.h
#  [build] FAILED: _deps/openxr-build/include/openxr/openxr_reflection.h /workspace/holoscan-openxr/build/_deps/openxr-build/include/openxr/openxr_reflection.h
RUN pip install markupsafe==2.0.1

# Copy vulkan sdk
# NOTE: It's all in x86_64 even if that's not the target platform
# (Vulkan SDK cmake scripting issue)
ARG VULKAN_SDK_VERSION
ENV VULKAN_SDK=/opt/vulkansdk/${VULKAN_SDK_VERSION}
COPY --from=vulkansdk-builder ${VULKAN_SDK}/x86_64/ ${VULKAN_SDK}
# We need to use the headers and shader compiler of the SDK but want to link against the
# Vulkan loader provided by the Ubuntu package. Therefore create a link in the SDK directory
# pointing to the system Vulkan loader library.
RUN rm -f ${VULKAN_SDK}/lib/libvulkan.so* \
    && ln -s /lib/$(uname -m)-linux-gnu/libvulkan.so.1 ${VULKAN_SDK}/lib/libvulkan.so
ENV PATH="${PATH}:${VULKAN_SDK}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${VULKAN_SDK}"

# enable CUDA cache
ENV CUDA_CACHE_DISABLE="0"

# all NVIDIA devices are visible, also use all driver capabilities
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="all"
