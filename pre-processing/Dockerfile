from nvidia/caffe

RUN apt-get update && apt-get install -y \
    python-dev \
    python-pip


RUN pip install pydicom
RUN apt-get install python-opencv



ARG CACHEBUST=0
WORKDIR / 
COPY requirements.txt DREAM_DM_preprocessing.py util.py preprocess.sh ./
RUN pip install -r requirements.txt
COPY Preprocessor ./Preprocessor
COPY config ./config
