FROM gcr.io/tensorflow/tensorflow:0.11.0rc2-gpu
MAINTAINER Minhwan Yu <minhwan90@gmail.com>

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \ 
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \ 
    libav-tools  \ 
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \ 
    libjasper-dev \ 
    libgtk2.0-dev \ 
    python-numpy \ 
    python-pycurl \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    python-opencv \ 
    qt5-default \
    libvtk6-dev \ 
    zlib1g-dev 

RUN git clone https://github.com/Itseez/opencv
RUN cd opencv
RUN git checkout tags/3.1.0

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_CUDA=OFF \
    -DINSTALL_C_EXAMPLES=OFF  \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -D WITH_TBB=ON ..
RUN make -j4
RUN make install

RUN /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig

RUN 'export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig' >> ~/.bashrc 
