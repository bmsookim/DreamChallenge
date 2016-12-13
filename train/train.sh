export netType='resnet'
export depth=50
export width=2
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'
export save=logs/${netType}-${depth}x${width}/
export experiment_number=1

th main.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0 \
-imageSize 224 \
-cropSize 224 \
-top5_display false \
-testOnly false \
-resume '' \
-depth ${depth} \
-widen_factor ${width}
