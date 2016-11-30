#!/bin/bash
nvidia-smi
python DREAM_DM_preprocessing.py -c dreamCh -d train -f class
