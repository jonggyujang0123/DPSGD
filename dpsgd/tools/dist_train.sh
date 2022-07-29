#!/bin/bash
CONFIG=$1
GPUS=$2
RESUME=${RESUME:-0}
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1:29500"}
torchrun --nnodes=$NNODES \
	--max_restarts=3 \
	--nproc_per_node=$GPUS \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_ADDR \
	tools/main.py \
	--config $CONFIG \
	--multigpu True
