FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu

ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && apt-get install -y libjpeg-dev libpng-dev libtiff5-dev libdc1394-22-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev && apt-get install -y libjsoncpp-dev && apt-get clean

RUN chmod +rwx /usr/bin/python3.8

# Install OpenCV
RUN git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git && \
 	cd opencv && mkdir -p build && cd build && \
 	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D WITH_TBB=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D WITH_CUBLAS=1 \
	-D WITH_CUDA=ON \
	-D BUILD_opencv_cudacodec=OFF \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D CUDA_ARCH_BIN=8.6 \
	-D WITH_V4L=ON \
	-D WITH_QT=OFF \
	-D WITH_OPENGL=ON \
	-D WITH_GSTREAMER=ON \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_PC_FILE_NAME=opencv.pc \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D PYTHON3_PACKAGES_PATH=/usr/lib/python3.8/dist-packages \
	-D PYTHON_EXECUTABLE=/usr/bin/python \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D BUILD_EXAMPLES=OFF .. && \
	make -j8 && \
	make install && \
	ldconfig