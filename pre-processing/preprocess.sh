#!/bin/bash
nvidia-smi

# workdir : ./pre-processing/
# image preprocessing
python DREAM_DM_preprocessing.py -c dreamCh -d train -f class -v  1 -p 6
#python DREAM_DM_preprocessing.py -c dreamCh -d train -f class -v 1 -p 4
#python DREAM_DM_preprocessing.py -c KUMC-guro -d train -f robust -p 4
#python DREAM_DM_preprocessing.py -c KUMC-guro -d train -f class -p 6 -v 1

# workdir : ./pre-processing/Torch
cd Torch

# build torch dataset 
th preprocess.lua

#workdir : ./
cd ../

# check result
RESULT_DIR='/preprocessedData'
echo "## Files in "$RESULT_DIR"/dreamCh"
ls $RESULT_DIR/dreamCh
echo "## Files in "$RESULT_DIR"/gen"
ls $RESULT_DIR/dreamCh/gen
