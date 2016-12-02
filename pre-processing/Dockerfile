from nvidia/caffe

RUN apt-get update && apt-get install -y \
    python-dev \
    python-pip \
    git \
    vim


RUN pip install pydicom

ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN git clone https://github.com/Itseez/opencv_contrib.git && \
    cd opencv_contrib && \
    git checkout 3.0.0

RUN git clone https://github.com/Itseez/opencv.git
WORKDIR /opencv
RUN git checkout 3.0.0
RUN mkdir build
WORKDIR /opencv/build

RUN apt-get install -y \
    cmake \
    build-essential
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_C_EXAMPLES=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON ..
RUN make -j $(nproc)

RUN \
    /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && \
    ldconfig && \
    echo  'export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig' >> ~/.bashrc && \
    cp /opencv/build/lib/cv2.so /usr/local/lib/python2.7/dist-packages/

WORKDIR /
RUN \ 
    rm -rf opencv && \
    rm -rf opencv_contrib

# install torch
RUN git clone https://github.com/torch/distro.git /torch --recursive
RUN apt-get install -y curl wget libreadline-dev
RUN cd torch && \
    ./install.sh

ARG CACHEBUST=0
WORKDIR / 
COPY requirements.txt  ./
RUN pip install -r requirements.txt
COPY DREAM_DM_preprocessing.py util.py preprocess.sh ./
COPY Preprocessor ./Preprocessor
COPY config ./config
COPY Torch ./Torch
