#!/bin/bash
export netType='resnet'
export depth=50
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'

ls /preprocessedData/dreamCh/train/0/* | wc -l
ls /preprocessedData/dreamCh/train/1/* | wc -l
ls /preprocessedData/dreamCh/val/0/* | wc -l
ls /preprocessedData/dreamCh/val/1/* | wc -l

rm -rf scratch/*

th main.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 128 \
    -LR 1e-3 \
    -weightDecay 1e-3 \
    -dropout 0 \
    -depth ${depth} \
    -retrain pretrained/resnet-${depth}.t7\
    -resetClassifier true \
    -nClasses 2 \
    -resume 'none' \
