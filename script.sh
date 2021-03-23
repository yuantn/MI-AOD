#!/usr/bin/env bash

CONFIG="configs/MIAOD.py"
GPUS=1
CUDA_NUM=$1
PORT=${PORT:-$((30000+$1*100))}

export CUDA_VISIBLE_DEVICES=$1
rm $(dirname "$0")/log_nohup/nohup_$1.log

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/tools/train.py $CONFIG --launcher pytorch ${@:3} > $(dirname "$0")/log_nohup/nohup_$1.log 2>&1 &
