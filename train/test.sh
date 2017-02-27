export netType='resnet'
export depth=18
export dataset='dreamChallenge'
export data='/preprocessedData'



th test.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -dropout 0 \
    -depth ${depth} \
    -resume '/modelState'
