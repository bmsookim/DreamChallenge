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


WORKDIR /root/torch
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

WORKDIR /
ARG CACHEBUST=0
COPY *.lua ./
COPY train.sh ./
COPY datasets ./datasets
COPY networks ./networks
