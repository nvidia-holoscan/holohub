# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============ Stage: base ============
# Configure the base environment
ARG FROM_IMAGE="holoscan-sdk-dev"
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.5.0 AS base

# End base image

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME
ENV TZ="America/Chicago"
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN apt-get update && \
    apt-get install -y wget vim libusb-1.0-0-dev libsndfile1 pkg-config iproute2 git-core cmake build-essential python3 python3-dev python3-pip python3-numpy portaudio19-dev python3-pyaudio swig iputils-ping rtl-sdr librtlsdr-dev net-tools openssh-client --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
 
WORKDIR /tmp

# cuSignal Setup
ENV CUSIGNAL_HOME=/tmp/cusignal
RUN cd /tmp && \
    git clone https://github.com/rapidsai/cusignal.git && \
    pip3 install cupy-cuda11x numba nvtx && \    
    pip3 install scipy>=1.5.0 boost matplotlib pytest pytest-benchmark sphinx pydata-sphinx-theme sphinx-copybutton numpydoc ipython && \
    pip3 install grpcio SoundFile PyYAML nvidia-riva-client
RUN cd $CUSIGNAL_HOME && \
    ./build.sh && \
    cd python && \
    pip install .
#end cuSignal

# SDR tool setup
RUN echo 'blacklist dvb_usb_rtl28xxu' > /etc/modprobe.d/blacklist-dvb_usb_rtl28xxu.conf && \
    apt-get install -y rtl-sdr librtlsdr-dev    

RUN git clone https://github.com/pothosware/SoapySDR.git && \
    cd SoapySDR && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    ldconfig

RUN cd /tmp && \
    git clone https://github.com/pothosware/SoapyRTLSDR.git && \
    cd SoapyRTLSDR && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

RUN cd /tmp && \
    git clone https://github.com/xmikos/simplesoapy.git && \
    cd simplesoapy && \
    pip3 install .
# End SDR tools


WORKDIR /workspace
