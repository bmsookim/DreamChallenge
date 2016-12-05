FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER MinhwanYu <minhwan90@gmail.com>

# install python(including pip), git, and vim
RUN apt-get update && apt-get install -y \
    python-dev \
    python-pip \
    git \
    vim

# set environment variables
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CUDA_HOME=/usr/local/cuda

# install opencv


# clone and setup py-faster-rcnn (from custom repo)



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
