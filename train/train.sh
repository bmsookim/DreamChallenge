#!/bin/bash
export netType='resnet'
export depth=50
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'

rm -rf scratch/*

th main.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 32 \
    -dropout 0.3 \
    -LR 1e-3 \
    -imageSize 224 \
    -depth ${depth} \
    -retrain pretrained/resnet-${depth}.t7\

cp -R /scratch/* /modelState/
