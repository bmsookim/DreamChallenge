#!/bin/bash

WORKSPACE=$PWD

# clone faster-rcnn
# clone caffe

if [ ! -d $WORKSPACE/py-faster-rcnn ]
then
    git clone https://github.com/rbgirshick/py-faster-rcnn.git
fi

if [ ! -d $WORKSPACE/py-faster-rcnn/caffe-fast-rcnn ]
then
    cd $WORKSPACE/py-faster-rcnn
    git clone https://github.com/BVLC/caffe.git ./caffe-fast-rcnn
    cp $WORKSPACE/installation/pycaffe.Makefile.config ./py-faster-rcnn/caffe-fast-rcnn/Makefile.config
fi

# install dependencies
cd $WORKSPACE/py-faster-rcnn/caffe-fast-rcnn/python
    # pip
sudo pip install -r requirements.txt
    # apt
sudo apt-get install -y \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libhdf5-serial-dev \
    protobuf-compiler\
    libgflags-dev libgoogle-glog-dev liblmdb-dev \
    libatlas-base-dev liblapack-dev && \
    apt-get install --no-install-recommends libboost-all-dev -y && \
# update configuration
ldconfig

# install libs
cd $WORKSPACE/py-faster-rcnn/lib && \
    make
# install caffe
cd $WORKSPACE/py-faster-rcnn/caffe-fast-rcnn && \
    make -j8 && make pycaffe
