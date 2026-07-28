#!/bin/bash
# HomeGrid + OCTSSM with SlotContrast object slots + one CNN image slot.
#
# SlotContrast extracts obs['slot'] (n_slots, dim) from the POV image.
# The model CNN encodes the same POV image into one global image slot. The text
# embedding is appended to every object/image slot. Object slots are decoded
# independently, while image and text are decoded from the image slot.
#
# Usage:
#   sh scripts/run_homegrid_masks_encoder.sh [GPU_ID] [SEED] [extra train.py args...]
#
# Examples:
#   sh scripts/run_homegrid_masks_encoder.sh 0 0
#   sh scripts/run_homegrid_masks_encoder.sh 0 0 --run.steps 1e6

set -euo pipefail

device="${1:-0}"
seed="${2:-0}"

shift 2 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$device"

SLOT_CFG="${SLOT_CFG:-/Users/a/code/slotcontrast_chp/homegrid_7slots_150k.yaml}"
SLOT_CKPT="${SLOT_CKPT:-/Users/a/code/slotcontrast_chp/homegrid_7slots_150k.ckpt}"

exec python dynalang/train.py \
  --configs homegrid slotcontrast cnn_image_text_slots octssm transformer_heads \
  --run.script train_custom_eval \
  --run.log_keys_video log_image \
  --logdir "data/slotcontrast_image_seed-${seed}" \
  --use_wandb False \
  --task homegrid_dynamics \
  --envs.amount 10 \
  --envs.eval_amount 10 \
  --run.eval_eps 66 \
  --seed "$seed" \
  --batch_size 16 \
  --batch_length 256 \
  --run.train_ratio 32 \
  --run.steps 50e6 \
  --dataset_excluded_keys info \
  --jax.mem_fraction 0.95 \
  --jax.profiler False \
  --slot_extractor.config_path "$SLOT_CFG" \
  --slot_extractor.checkpoint_path "$SLOT_CKPT" \
  --octssm.action_mode slot \
  --run.eval_every 250000 \
  "$@"
