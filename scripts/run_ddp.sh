#!/bin/bash

NPROC_PER_NODE=2
MASTER_PORT=22222
TRAIN_SCRIPT=cifar10_example_ddp.py
TRAIN_SCRIPT_STEM=`echo $TRAIN_SCRIPT | sed 's/\.[^\.]*$//'`

RUN_COMMAND="nohup python -u -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT $TRAIN_SCRIPT > $TRAIN_SCRIPT_STEM.log 2> $TRAIN_SCRIPT_STEM.err&"
echo $RUN_COMMAND
eval $RUN_COMMAND