#!/bin/bash

NNODES=2
NODE_RANK=1
NPROC_PER_NODE=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=22222
TRAIN_SCRIPT=cifar10_example_ddp.py
TRAIN_SCRIPT_STEM=`echo $TRAIN_SCRIPT | sed 's/\.[^\.]*$//'`

RUN_COMMAND="nohup python -u -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NPROC_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $TRAIN_SCRIPT > $TRAIN_SCRIPT_STEM-$NODE_RANK.log 2> $TRAIN_SCRIPT_STEM-$NODE_RANK.err&"
echo $RUN_COMMAND
eval $RUN_COMMAND
