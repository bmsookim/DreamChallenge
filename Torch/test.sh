export netType='wide-resnet'
export depth=10
export width=1
export dataset='dreamChallenge'
export data='preprocessedData'

th test.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0 \
-imageSize 512 \
-cropSize 512 \
-top5_display false \
-depth ${depth} \
-widen_factor ${width}

