FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER MinhwanYu <minhwan90@gmail.com>

# set environment variables
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# install dependencies
RUN apt-get update
RUN apt-get install -y \
	git \
	wget

# install torch
WORKDIR /opt
RUN git clone https://github.com/torch/distro.git /root/torch --recursive
RUN cd /root/torch && \
    bash install-deps && \
    ./install.sh

RUN /root/torch/install/bin/luarocks install nn      && \
    /root/torch/install/bin/luarocks install optnet  && \
    /root/torch/install/bin/luarocks install cunn    && \
    /root/torch/install/bin/luarocks install cudnn   && \
    /root/torch/install/bin/luarocks install cutorch && \
    /root/torch/install/bin/luarocks install image   
