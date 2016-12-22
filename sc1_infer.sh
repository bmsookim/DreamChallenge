#!/bin/bash
WORKDIR=$(pwd)

# remove previous preprocessed test data
ls /
pwd
ls /preprocssedData
ls /output


# inference 
cd $WORKDIR/train
python test.py -q test -c dreamCh -d test
