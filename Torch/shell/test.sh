export netType='wide-resnet'
export dataset='dreamChallenge'
export data='preprocessedData'

th main.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0.5 \
-resume 'scratch' \
-testPhase true \
-top5_display false \
-depth 28 \
-widen_factor 10

