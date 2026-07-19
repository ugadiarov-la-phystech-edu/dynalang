#! /bin/bash
# Train dynalang on HomeGrid with the in-graph SlotContrast extractor trained
# FROM SCRATCH: frozen pretrained DINO backbone (timm hub weights), randomly
# initialized slot attention / projection / learned init / featrec MLPDecoder.
# The architecture (7 slots, DINOv2-S/14 @ 336) comes from the reference
# SlotContrast config; the fresh-init pickle is generated on first use (one
# per seed, so the extractor init is seed-controlled).
#
# Usage: sh scripts/run_homegrid_jax_slots_scratch.sh <task> <name> <gpu> <seed> [extra --flags...]
# e.g.:  sh scripts/run_homegrid_jax_slots_scratch.sh homegrid_task scratch 0 0

task=$1
name=$2
device=$3
seed=$4

shift
shift
shift
shift

# Reference SlotContrast settings yaml (architecture only, no weights used).
sc_config=${SC_CONFIG:-/home/ugadiarov/slotcontrast_ckpt/homegrid/homegrid_7slots_150k.yaml}
fresh_pkl=checkpoints/slotcontrast_fresh_homegrid7_seed${seed}_jax.pkl

export COMET_API_KEY=g4L2fxT5u66seUlbZYZrsQPce
export COMET_PROJECT_NAME=homegrid
export COMET_EXPERIMENT_NAME=${name}_${seed}
#export COMET_RUN_ID=

path=logdir/homegrid/${COMET_EXPERIMENT_NAME}
mkdir -p ${path}

if [ ! -f ${fresh_pkl} ]; then
  python scripts/convert_slotcontrast_to_jax.py --fresh \
    --config ${sc_config} --output ${fresh_pkl} --seed ${seed} || exit 1
fi

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --configs homegrid_octssm_jax_slots homegrid_octssm_jax_slots_scratch \
  --run.script train_custom_eval \
  --logdir ${path} \
  --task $task \
  --seed $seed \
  --use_wandb False \
  --encoder.slotcontrast.n_slots 7 \
  --encoder.slotcontrast.jax_checkpoint ${fresh_pkl} \
  --encoder.slotcontrast.chunk 512 \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --envs.amount 66 \
  --envs.eval_amount 11 \
  --run.eval_eps 66 \
  --jax.mem_fraction 0.95 \
  --jax.profiler False \
  "$@" &>${path}/log.txt
