export netType='wide-resnet'
export dataset='dreamChallenge'
export data='preprocessedData'
export save=logs/${netType}
export experiment_number=1
mkdir -p $save
mkdir -p modelState

rm -rf modelState/gen/dreamChallenge.t7

th main.lua \
-dataset ${dataset} \
-data ${data} \
-netType ${netType} \
-nGPU 2 \
-batchSize 32 \
-top5_display false \
-depth 28 \
-widen_factor 10 \
| tee $save/train_log${experiment_number}.txt
