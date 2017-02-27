#!/bin/bash
export netType='resnet'
export depth=18
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'

rm -rf scratch/*

th main.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 128 \
    -LR 1e-3 \
    -weightDecay 5e-3 \
    -dropout 0.3 \
    -depth ${depth} \
    -retrain pretrain/resnet-${depth}.t7\
    -resetClassifier true \
    -nClasses 2 \
    -resume 'none' \

cp -R /scratch/* /modelState/
