#!/bin/bash
export netType='wide-resnet'
export depth=34
export width=4
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'

rm -rf scratch/*

th main.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-batchSize 32 \
-dropout 0.3 \
-LR 1e-2 \
-imageSize 224 \
-depth ${depth} \
-widen_factor ${width} \ 

# cp -R /scratch/* /modelState/
