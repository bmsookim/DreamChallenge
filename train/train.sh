#!/bin/bash
export netType='wide-resnet'
export depth=40
export width=2
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
-LR 1e-2 \
-imageSize 256 \
-cropSize 256 \
-saveCut false \
-top5_display false \
-testOnly false \
-depth ${depth} \
-widen_factor ${width} 

# cp -R /scratch/* /modelState/
