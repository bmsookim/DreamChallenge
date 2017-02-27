#!/bin/bash
WORKDIR=$(pwd)

ls /modelState
# inference 
cd $WORKDIR/train
python test.py \
    --config ../dicom-preprocessing/config/train.yaml   \
    --corpus dreamCh    \
    --dataset test      \
    --exam_meta False   \
    --processor 1       \
    --log False         \
    --form mem          \
    --sampler default
