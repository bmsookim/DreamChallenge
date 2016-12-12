#!/bin/bash

WORKSPACE=$PWD

# clone faster-rcnn
# clone caffe
git clone https://github.com/rbgirshick/py-faster-rcnn.git

cd $WORKSPACE/py-faster-rcnn
git clone https://github.com/BVLC/caffe.git ./caffe-fast-rcnn
cp $WORKSPACE/installation/pycaffe.Makefile.config $WORKSPACE/py-faster-rcnn/caffe-fast-rcnn/Makefile.config

# install dependencies
cd $WORKSPACE/py-faster-rcnn/caffe-fast-rcnn/python

# install libs
cd $WORKSPACE/py-faster-rcnn/lib && \
    make
# install caffe
cd $WORKSPACE/py-faster-rcnn/caffe-fast-rcnn && \
    make -j8 && make pycaffe

# get trained model
cd $WORKSPACE
wget http://infos.korea.ac.kr/crawl/model.tar.gz
tar -zxvf model.tar.gz
