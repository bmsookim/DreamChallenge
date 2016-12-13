export netType='wide-resnet'
export depth=50
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
-imageSize 224 \
-cropSize 224 \
-top5_display false \
-depth ${depth} \
-widen_factor ${width}

