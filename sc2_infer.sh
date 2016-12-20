#!/bin/bash
WORKDIR=$(pwd)

# remove previous preprocessed test data
rm -rf /preprocssedData/test

# image preprocessing
cd $WORKDIR/pre-processing
python DREAM_DM_preprocessing.py --corpus dreamCh --dataset test --form robust --processor 6 --queue test --exams_metada 0

# inference 
cd $WORKDIR/train
./test.sh

# scoring
cd $WORKDIR/test
python score.py
