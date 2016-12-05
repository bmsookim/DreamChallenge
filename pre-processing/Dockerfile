FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER MinhwanYu <minhwan90@gmail.com>

# install packages
RUN apt-get update && apt-get install -y \
    python-dev \
    python-pip \
    git \
    vim \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    libtiff5-dev \
    libjpeg-dev \
    libjasper-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine-dev \
    libgstreamer0.10-dev \
    libgstreamer-plugins-base0.10-dev \
    libv4l-dev \
    python-dev \
    python-numpy \
    python-pip \
    libtbb-dev \
    libeigen3-dev \
    libqt4-dev \
    libgtk2.0-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    x264 \ 
    v4l-utils \
    libgtk2.0-dev \
    unzip \
    wget \
    curl \
    libreadline-dev

# set environment variables
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# install opencv
WORKDIR /opt
RUN git clone https://github.com/Itseez/opencv.git && \
    cd /opt/opencv &&\
    git checkout 3.0.0
RUN git clone https://github.com/Itseez/opencv_contrib.git && \
    cd /opt/opencv_contrib && \
    git checkout 3.0.0

WORKDIR /opt/opencv
RUN mkdir build
WORKDIR /opt/opencv/build
RUN pip install numpy
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_C_EXAMPLES=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON .. \
        -DCUDA_CUDA_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcudart.so && \
    make -j$(nproc) && install && \
    echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf && \
    ldconfig


# clone and setup py-faster-rcnn (from custom repo)
WORKDIR /opt
RUN git clone https://github.com/torch/distro.git /torch --recursive
RUN cd torch && \
    ./install.sh


# copy preprocessing project
WORKDIR / 
    # install python dependencies
COPY requirements.txt  ./
RUN pip install -r requirements.txt
    # end-point execution files
COPY DREAM_DM_preprocessing.py util.py preprocess.sh ./
    # Preprocessor module
COPY Preprocessor ./Preprocessor
    # configurations (logging, preprocessing)
COPY config ./config
    # Torch dataset builder
COPY Torch ./Torch
