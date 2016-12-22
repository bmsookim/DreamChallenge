#!/bin/bash
WORKDIR=$(pwd)

# inference 
cd $WORKDIR/train
python test.py -q test -c dreamCh -d test

