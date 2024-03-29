ARG BASE_IMAGE=holohub:ngc-v1.0.3-dgpu
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && \
    apt install -y \
        libglvnd-dev \
        ninja-build

# Create directories
RUN mkdir -p /tmp/vtk/ && \
    mkdir -p /opt/vtk/

WORKDIR /tmp/vtk
RUN curl --remote-name https://gitlab.kitware.com/vtk/vtk/-/archive/v9.3.0/vtk-v9.3.0.tar.gz && \
    tar -xvzf vtk-v9.3.0.tar.gz && \
    rm vtk-v9.3.0.tar.gz && \
    cmake -GNinja -S vtk-v9.3.0 -B vtk-build \
        -DVTK_WRAP_PYTHON=ON \
        -DVTK_MODULE_ENABLE_RenderingCore=YES \
        -DVTK_MODULE_ENABLE_RenderingFFMPEGOpenGL2=YES \
        -DVTK_MODULE_ENABLE_RenderingOpenGL2=YES \
        -DVTK_MODULE_ENABLE_Python=YES && \
    cmake --build vtk-build && \
    cmake --install vtk-build --prefix=/opt/vtk && \
    rm -rf vtk-build
