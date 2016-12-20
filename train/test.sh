export netType='convnet'
export depth=34
export width=2
export dataset='dreamChallenge'
export data='/preprocessedData'

th test.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0 \
-imageSize 256 \
-cropSize 256 \
-top5_display false \
-depth ${depth} \
-widen_factor ${width}

