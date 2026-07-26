#! /bin/bash
# Train dynalang on HomeGrid with the in-graph SlotContrast extractor
# initialized from a PRE-TRAINED SlotContrast checkpoint (fine-tuning: frozen
# DINO backbone, slot attention / projection / init + featrec MLPDecoder start
# from the trained weights). The torch checkpoint is converted to a ninjax
# pickle on first use; the pickle is deterministic, so one file serves all
# seeds.
#
# Usage: sh scripts/run_homegrid_jax_slots_pretrained.sh <task> <name> <gpu> <seed> [extra --flags...]
# e.g.:  sh scripts/run_homegrid_jax_slots_pretrained.sh homegrid_task pretrained 0 0

task=$1
name=$2
device=$3
seed=$4

shift
shift
shift
shift

# Trained SlotContrast checkpoint + its settings yaml.
sc_config=${SC_CONFIG:-/home/ugadiarov/slotcontrast_ckpt/homegrid/homegrid_7slots_150k.yaml}
sc_ckpt=${SC_CKPT:-/home/ugadiarov/slotcontrast_ckpt/homegrid/homegrid_7slots_150k.ckpt}
pkl=checkpoints/slotcontrast_homegrid7_150k_pred_jax.pkl

export COMET_API_KEY=g4L2fxT5u66seUlbZYZrsQPce
export COMET_PROJECT_NAME=homegrid
export COMET_EXPERIMENT_NAME=${name}_${seed}
#export COMET_RUN_ID=

path=logdir/homegrid/${COMET_EXPERIMENT_NAME}
mkdir -p ${path}

if [ ! -f ${pkl} ]; then
  python scripts/convert_slotcontrast_to_jax.py \
    --checkpoint ${sc_ckpt} --config ${sc_config} --output ${pkl} || exit 1
fi

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --configs homegrid_octssm_jax_slots \
  --run.script train_custom_eval \
  --logdir ${path} \
  --task $task \
  --seed $seed \
  --use_wandb False \
  --encoder.slotcontrast.n_slots 7 \
  --encoder.slotcontrast.jax_checkpoint ${pkl} \
  --encoder.slotcontrast.chunk 512 \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --envs.amount 66 \
  --envs.eval_amount 11 \
  --run.eval_eps 66 \
  --jax.mem_fraction 0.95 \
  --jax.profiler False \
  "$@" &>${path}/log.txt
