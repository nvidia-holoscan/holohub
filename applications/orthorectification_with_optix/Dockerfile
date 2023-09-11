# syntax = docker/dockerfile:1.2

ARG FROM_IMAGE="nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"

# ============ Stage: base ============
FROM ${FROM_IMAGE} AS base

# Configure the base conda environment
ARG CONDA_ENV_NAME=ortho_holoscan
ARG PYTHON_VER=3.8

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Install libarchive
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y libarchive13

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash
RUN ln -sf /bin/bash /bin/sh

# Install mamba to speed the solve up
RUN conda config --set ssl_verify false &&\
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    conda config --env --add channels conda-forge &&\
    /opt/conda/bin/conda install -y -n base -c conda-forge "mamba >=0.22" "boa >=0.10" python=${PYTHON_VER}

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n ${CONDA_ENV_NAME} -c conda-forge python=${PYTHON_VER}

RUN source activate ${CONDA_ENV_NAME} && mamba install -c conda-forge -y numpy cupy pillow pytest gdal shapely && \
    pip install pynvrtc tqdm opencv-python odm_report_shot_coverage

# MUST RUN BUILD COMMAND FROM ABOVE HOLOHUB-INTERNAL REPO
WORKDIR /work
COPY ./NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64 /work/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64

RUN git clone https://github.com/NVIDIA/optix-toolkit.git
WORKDIR /work/optix-toolkit
RUN git checkout v0.8.1
RUN git submodule update --init --recursive PyOptiX
WORKDIR /work/optix-toolkit/PyOptiX/optix

RUN source activate ${CONDA_ENV_NAME} &&\
    export PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=/work/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64" &&\
    export PYOPTIX_STDDEF_DIR="/usr/include/linux" &&\
    python setup.py install
WORKDIR /work

# now install holoscan python bindings
RUN source activate ${CONDA_ENV_NAME} && pip install holoscan
WORKDIR /work

RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc