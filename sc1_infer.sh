#!/bin/bash
WORKDIR=$(pwd)

ls /modelState
# inference 
cd $WORKDIR/train
python test.py -q test -c dreamCh -d test
