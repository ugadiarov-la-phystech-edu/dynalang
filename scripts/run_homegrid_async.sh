#! /bin/bash

task=$1
name=$2
device=$3
seed=$4

shift
shift
shift
shift

export COMET_API_KEY=
export COMET_PROJECT_NAME=homegrid
export COMET_EXPERIMENT_NAME=${name}_${seed}
export COMET_RUN_ID=

path=logdir/homegrid/${COMET_EXPERIMENT_NAME}
mkdir -p ${path}

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --configs xlarge \
  --run.script train_custom_eval \
  --logdir ${path} \
  --use_wandb False \
  --task $task \
  --envs.amount 66 \
  --seed $seed \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot \
  --batch_size 16 \
  --batch_length 256 \
  --jax.mem_fraction 0.95 \
  --jax.profiler False \
  --run.log_every 300 \
  --envs.eval_amount 11 \
  --run.eval_eps 66 \
  --run.eval_every 250000 \
  --run.train_ratio 32 &>${path}/log.txt &
