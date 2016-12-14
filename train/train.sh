export netType='resnet'
export depth=50
export width=1
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'

th main.lua \
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
-resume '' \
-depth ${depth} \
-widen_factor ${width}
