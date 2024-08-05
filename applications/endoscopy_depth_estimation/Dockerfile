ARG BASE_IMAGE
FROM ${BASE_IMAGE} as base

ARG OPEN_CV_VERSION=4.10.0
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


RUN apt-get update && apt-get install -y build-essential cmake git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff5-dev libdc1394-dev libeigen3-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm \
    libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libjsoncpp-dev ffmpeg \
     && apt-get clean

RUN chmod +rwx /usr/bin/python3.10

# Install OpenCV
RUN git clone --depth 1 --branch $OPEN_CV_VERSION https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch $OPEN_CV_VERSION https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_CUDA=ON \
    -D BUILD_opencv_cudacodec=OFF \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=8.6,8.7,8.9 \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3.10/dist-packages \
    -D PYTHON_EXECUTABLE=/usr/bin/python \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    .. && \
    make -j8 && \
    make install && \
    ldconfig
