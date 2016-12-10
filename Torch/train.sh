export netType='wide-resnet'
export depth=10
export width=1
export dataset='dreamChallenge'
export data='/preprocessedData/dreamCh/'
export save=logs/${netType}-${depth}x${width}/
export experiment_number=1
export CUDA_VISIBLE_DEVICES="0,1"
mkdir -p $save
mkdir -p modelState
mkdir -p scratch

th main.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0 \
-imageSize 1024 \
-cropSize 1024 \
-top5_display false \
-testOnly false \
-resume '' \
-depth ${depth} \
-widen_factor ${width} \
# | tee $save/train_log${experiment_number}.txt
