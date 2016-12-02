#!/bin/bash
nvidia-smi

# workdir : ./pre-processing/
cd pre-processing/

# image preprocessing
python DREAM_DM_preprocessing.py -c dreamCh -d train -f class -v  1

# workdir : ./pre-processing/Torch
cd Torch

# build torch dataset 
th preprocess.lua

#workdir : ./
cd ../../
