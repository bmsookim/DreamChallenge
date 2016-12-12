export netType='wide-resnet'
export depth=52
export width=4
export dataset='dreamChallenge'
export data='preprocessedData'
export CUDA_VISIBLE_DEVICES="0,1"

th test.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0 \
-imageSize 1024 \
-cropSize 1024 \
-top5_display false \
-depth ${depth} \
-widen_factor ${width}

