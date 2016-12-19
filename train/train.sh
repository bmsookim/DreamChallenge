#!/bin/bash
export netType='wide-resnet'
export depth=10
export width=1
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'

th main.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0.3 \
-imageSize 256 \
-cropSize 256 \
-saveCut true \
-top5_display false \
-testOnly false \
-depth ${depth} \
-widen_factor ${width} \

#th convert.lua \
#-dataset ${dataset} \
#-data ${data} \
#-netType ${netType} \
#-nGPU 2 \
#-batchSize 32 \
#-dropout 0 \
#-imageSize 512 \
#-cropSize 512 \
#-top5_display false \
#-testOnly false \
#-depth ${depth} \
#-widen_factor ${width} \

#mv /scratch/* /modelState/ 
