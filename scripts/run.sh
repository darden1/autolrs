#!/bin/bash

TRAIN_SCRIPT=cifar10_example.py
TRAIN_SCRIPT_STEM=`echo $TRAIN_SCRIPT | sed 's/\.[^\.]*$//'`

RUN_COMMAND="nohup python -u $TRAIN_SCRIPT > $TRAIN_SCRIPT_STEM.log 2> $TRAIN_SCRIPT_STEM.err&"
echo $RUN_COMMAND
eval $RUN_COMMAND
