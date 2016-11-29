FROM nvidia/cuda:8.0-runtime
MAINTAINER Minhwan Yu <minhwan90@gmail.com>

RUN apt-get update && apt-get install -y \
    curl && \
    rm -rf /var/lib/apt/lists/ && \
    apt-get update && \
    apt-get install build-essential cmake git pkg-config -y && \
    apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev -y && \
    apt-get install libgtk2.0-dev -y && \
    apt-get install python2.7-dev -y

RUN \
    CUDNN_DOWNLOAD_SUM=a87cb2df2e5e7cc0a05e266734e679ee1a2fadad6f06af82a76ed81a23b102c8 && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict - && \
    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local --wildcards 'cuda/lib64/libcudnn.so.*' && \
    rm cudnn-8.0-linux-x64-v5.1.tgz && \
    ldconfig

# install tensorflow
RUN \
    apt-get install python-pip -y && \
    pip install --upgrade pip && \
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl && \
    pip install --upgrade $TF_BINARY_URL

# update environment variables
RUN \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.4 /usr/lib/x86_64-linux-gnu/libcudnn.so && \
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} && \
    export CUDA_HOME=/usr/local/nvidia/

# install opencv
RUN \
    cd ~ && \
    git clone https://github.com/Itseez/opencv.git && \
    cd opencv && \
    git checkout 3.0.0 && \
    cd ~ && \
    git clone https://github.com/Itseez/opencv_contrib.git && \
    cd opencv_contrib && \
    git checkout 3.0.0 && \
    cd /root/opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_C_EXAMPLES=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON .. && \
    cd ~/opencv/build && \
    make -j $(nproc) && \
    make install && \
    /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && \
    ldconfig && \
    echo  'export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig' >> ~/.bashrc && \
    cp ~/opencv/build/lib/cv2.so /usr/local/lib/python2.7/dist-packages/ && \
    rm -rf ~/opencv && \
    rm -rf ~/opencv_contrib

ARG CACHEBUST=0
WORKDIR / 
COPY train.sh ./
