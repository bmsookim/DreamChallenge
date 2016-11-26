#!/bin/bash

PREPROCESS_DIRECTORY="/preprocessedData"

echo "Install dependencies"


echo "execute"
cd pre-processing
python preprocessing.py -c dreamCh -d pilot 
