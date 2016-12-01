FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER contact@invasis.com

# Install base tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    dh-autoreconf

# Install build & doc tools
RUN apt-get install -y --no-install-recommends \
    cmake \
    doxygen

# Install basic cli tools
RUN apt-get install -y --no-install-recommends \
    unzip \
    git \
    wget

# install libs
RUN apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libprotobuf-dev \
    libsnappy-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdc1394-22-dev \
    libxine2-dev \
    zlib1g-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libgstreamer0.10-dev \
    libgstreamer-plugins-base0.10-dev \
    gstreamer-tools \
    libv4l-dev \
    v4l-utils \
    libgdal-dev \
    libeigen3-dev \
    libcurl3-dev \
    x264 \
    x265 \
    protobuf-compiler \
    swig

# Install Python 2 & 3
RUN apt-get install -y --no-install-recommends \
    python-dev \
    python-pip \
    python-virtualenv \
    python-wheel

# Install Java 8 JDK
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:webupd8team/java
RUN echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
RUN echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections

RUN apt-get update && apt-get install -y --no-install-recommends \
    oracle-java8-installer

RUN apt-get install -y --no-install-recommends \
    oracle-java8-set-default

ENV JAVA_HOME "/usr/lib/jvm/java-8-oracle"

# CLEAN
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenCV 3.1
ENV OPENCV_ROOT /opt/opencv-3.1.0
WORKDIR $OPENCV_ROOT

RUN git clone https://github.com/opencv/opencv.git .
RUN git checkout de35c59ba4cf863d014e130c9116a0d5dade7c91
WORKDIR $OPENCV_ROOT/build
RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D CUDA_FAST_MATH:BOOL="1" -D WITH_NVCUVID:BOOL="1" -D WITH_CUBLAS:BOOL="1" -D BUILD_DOCS:BOOL="1" ..
RUN make -j7
RUN make install && echo "/usr/local/lib" | tee -a /etc/ld.so.conf.d/opencv.conf && ldconfig
RUN cp /opt/opencv-3.1.0/build/lib/cv2.so /usr/lib/python2.7/dist-packages/cv2.so

# COPY Dream Challenge Source Code
ARG CACHEBUST=0
WORKDIR / 
COPY requirements.txt DREAM_DM_preprocessing.py util.py preprocess.sh ./
RUN pip install -r requirements.txt
COPY Preprocessor ./Preprocessor
COPY config ./config
