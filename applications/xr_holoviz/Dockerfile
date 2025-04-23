
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS run

ARG DEBIAN_FRONTEND=noninteractive

# Install Vulkan SDK and ICD.
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | gpg --dearmor -o /usr/share/keyrings/lunarg-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/lunarg-archive-keyring.gpg] https://packages.lunarg.com/vulkan/1.3.204  jammy main" \
    | tee /etc/apt/sources.list.d/lunarg-vulkan-1.3.204 -jammy.list
RUN apt update \
    && apt install --no-install-recommends -y \
        vulkan-icd \
        vulkan-sdk

# Install OpenXR SDK Dependencies.
RUN apt update \
    && apt install --no-install-recommends -y \
        build-essential \
        cmake \
        libeigen3-dev \
        libgl1-mesa-dev \
        libvulkan-dev \
        libwayland-dev\
        libx11-xcb-dev \
        libxcb-dri2-0-dev \
        libxcb-glx0-dev \
        libxcb-icccm4-dev \
        libxcb-keysyms1-dev \
        libxcb-randr0-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        mesa-common-dev \ 
        libopenxr-loader1 \ 
        libopenxr-dev \
        libopenxr1-monado 

# Install Python builder.
RUN pip install --upgrade pip
RUN pip install build

# Set the working directory
WORKDIR /app/xr_holoviz

#
# Holohub run setup
#
RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities/holohub_autocomplete /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup
