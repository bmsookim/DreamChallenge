export netType='resnet'
export depth=50
export dataset='dreamChallenge'
export data='preprocessedData'
export save=logs/${netType}-${depth}
export experiment_number=1
mkdir -p $save
mkdir -p modelState
mkdir -p scratch

th main.lua \
-testPhase false \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-dropout 0.3 \
-top5_display false \
-testOnly false \
-resume '' \
-depth ${depth} \
-widen_factor 2 \
# | tee $save/train_log${experiment_number}.txt
