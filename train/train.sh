#!/bin/bash
export netType='convnet'
export depth=34
export width=4
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'

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

th convert.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0 \
-imageSize 512 \
-cropSize 512 \
-top5_display false \
-testOnly false \
-depth ${depth} \
-widen_factor ${width} 

tar -cvzf /modelState/result.tar.gz /scratch/*
