# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dynalang is a reinforcement learning agent that uses language to model the world, based on the paper "Learning to Model the World with Language." It extends DreamerV3 with multimodal (language + vision) world modeling. The agent learns to predict future states using both visual observations and diverse language inputs (task descriptions, corrections, future predictions, dynamics info).

## Key Commands

### Install
```bash
pip install -e .            # or: pip install -e ".[cuda]" for the JAX CUDA 12 plugin
```
`install.txt` has the exact pinned versions that are known to work (Python 3.11, `jax[cuda12]==0.4.38`, `homegrid==0.1.1`, etc.). For GPU training you also need the CUDA compiler toolkit (`pip install nvidia-cuda-nvcc-cu12`); if JAX reports `ptxas not found` / `libdevice not found`, set `XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"`. Verify with `python -c "import jax; print(jax.devices())"`.

### Tests
The forked `embodied` library carries pytest tests (no project-level pytest config):
```bash
pytest dynalang/embodied/tests/                       # all
pytest dynalang/embodied/tests/test_replay.py         # one file
pytest dynalang/embodied/tests/test_replay.py::test_name   # one test
```

### Training (main entry point)
```bash
python dynalang/train.py --configs <config_presets> [--flag value ...]
```

Example training runs use shell scripts in `scripts/`:
```bash
sh scripts/run_homegrid.sh <task_name> <exp_name> <gpu_ids> <seed>
sh scripts/run_homegrid_transformer.sh <task_name> <exp_name> <gpu_ids> <seed>   # tssm world model
sh scripts/run_messenger_s1.sh <exp_name> <gpu_ids> <seed>
sh scripts/run_vln.sh <exp_name> <gpu_ids> <seed>
```

### Run scripts (`--run.script`)
`train.py` dispatches to a loop in `dynalang/embodied/run/`. Besides the DreamerV3 defaults (`train`, `train_eval`, `parallel`, `eval_only`, `offline`), this fork adds `train_custom_eval` (`train_cusom_eval.py` — note the filename typo), used by the HomeGrid transformer scripts for periodic multi-episode eval.

### Debug mode
Use `--configs defaults debug <env>` to run on CPU with small model, fast logging, and single environment.

### Multi-GPU
Paper batch sizes generally need more than one GPU. Pass comma-separated GPU IDs and split them between world-model training and policy:
```bash
--jax.train_devices 0,1,2,3 --jax.policy_devices 0
```
For throughput, use the `parallel` run script (env stepping and agent training in separate processes) — see `scripts/run_vln.sh`.

## Architecture

### Core Training Loop
- **`dynalang/train.py`** — Entry point. Parses config, creates env/agent/replay, dispatches to a run script (`train`, `train_eval`, `parallel`, `offline`, etc.).
- **`dynalang/embodied/run/`** — Training loop implementations. `train.py` is the basic loop; `parallel.py` splits env stepping and agent training across processes.

### Agent Structure (`dynalang/agent.py`)
The `Agent` class (decorated with `@jaxagent.Wrapper`) contains:
- **WorldModel** — encoder + RSSM dynamics model + decoder heads (reward, continue, observation reconstruction)
- **Task behavior** — actor-critic policy (default: `Greedy` from `behaviors.py`)
- **Exploration behavior** — optional separate exploration policy

### World Model Variants
Configured via `rssm_type` in `configs.yaml`; dispatched in `Agent.__init__` (`dynalang/agent.py` ~line 171). Each has its own config block in `configs.yaml` (`rssm`, `early_rssm`, `token_rssm`, `tssm`, `octssm`):
- `rssm` — Standard GRU-based RSSM (default). Optional `impl: maskgit` swaps stochastic sampling for a MaskGit head.
- `early` — `EarlyRSSM`: language fused before the GRU.
- `token` — `TokenRSSM`: vocabulary-based token prediction.
- `tssm` — `TSSM`: spatio-temporal Transformer replaces the GRU (`dynalang/transformer.py`). A causal Transformer attends over the last `tf_context_length` timesteps instead of recurring.
- `octssm` — `ObjectCentricTSSM` (subclass of `TSSM`): per-slot latents, one Transformer track per object slot plus cross-slot attention. When the encoder produces object slots, `octssm.num_slots` is auto-set from `encoder.n_output_slots`. State/head tensors gain a slot axis, so `octssm` uses different `head_dims`/`mlp_dims` (4/3 instead of `'deter'`) throughout `agent.py`.

All world models are in `dynalang/nets.py`; Transformer building blocks are in `dynalang/transformer.py` (ninjax re-implementation of the `torch.nn.Transformer` API, batch-first, PyTorch mask conventions).

### Object-Centric / Slot Pipeline
Recent work (branch `masks_encoder`) adds object-centric world modeling. NOTE: this pipeline differs per branch — `masks_encoder` has `slot_image`/`n_output_slots` in `MultiEncoder`; `one-attn` has a simpler slot passthrough plus the in-graph JAX extractor below. Slots reach the world model three ways:
- **In-graph JAX SlotContrast (trainable)** — `encoder.slotcontrast.enabled: True` embeds a ninjax re-implementation of SlotContrast (`dynalang/slot_nets.py`: timm-ViT backbone, slot attention, recurrent slot init reset at `is_first`) inside `MultiEncoder`, consuming the raw `image` key. Torch checkpoints are converted with `scripts/convert_slotcontrast_to_jax.py` (mapping in `dynalang/slot_convert.py`) and loaded via `encoder.slotcontrast.jax_checkpoint`. Freeze/fine-tune via `model_opt.frozen_keys` (e.g. `'enc/slotcontrast/backbone'`). The decoder reconstructs `sg(slots)` as a distillation target. Presets: `homegrid_octssm_jax_slots`(+`_debug`). Parity tests: `pytest tests/` (see `tests/test_real_checkpoint.py` for verifying a converted checkpoint; needs `SLOTCONTRAST_CKPT`/`SLOTCONTRAST_CFG` env vars).
- **External slot extractor** — `use_slot_extractor: True` wraps the batch env in `BatchSlotExtractorEnv` (`dynalang/embodied/core/slot_batch_env.py`), which runs a pretrained slot model (e.g. SlotContrast) on each frame and emits a `slot` observation key. Extractors subclass `SlotExtractor` (`dynalang/embodied/core/slot_extractor.py`); wired up in `make_slot_extractor` (`dynalang/train.py`). See the `slotcontrast` config preset (needs external `config_path`/`checkpoint_path`).
- **HomeGrid ground-truth object slots** — `homegrid.use_object_slots: True` adds the `SlotImageWrapper` (`dynalang/embodied/envs/homegrid.py`) that uses the sim's instance-segmentation mask to emit a `slot_image` tensor `(num_slots, H, W, 3)`: one per-object masked RGB image (only that object visible, rest black), grouped into fixed semantic slots (scene/agent/fixtures/trashcan/4 trash types) with a stable per-episode slot index. `MultiEncoder` (`dynalang/nets.py`) detects `slot`/`slot_image` keys and outputs a `(..., n_output_slots, dim)` slot tensor consumed by `octssm`.

### JAX Framework
- **`dynalang/ninjax.py`** — Custom stateful module system for JAX (like a lightweight Flax/Haiku). Modules use `self.get()` for parameters and `nj.rng()` for random keys. Global `CONTEXT` dict holds state during execution.
- **`dynalang/jaxagent.py`** — `JAXAgent` wrapper that handles JIT compilation, device placement, multi-device sharding, and async train/policy execution.
- **`dynalang/jaxutils.py`** — JAX utilities (custom distributions, scan helpers, optimizers).

### Environment Integration
- **`dynalang/embodied/`** — Forked from DreamerV3's `embodied` library. Provides env wrappers, replay buffers, logging, config system, and run loops.
- **`dynalang/embodied/envs/`** — Environment adapters. Each file wraps a specific env (HomeGrid, Messenger, VLN, LangRoom, Atari, DMC, etc.) into the common `embodied.Env` interface.
- Task names follow `suite_taskname` format (e.g., `homegrid_task`, `vln_default`, `messenger_s2`).

### Configuration System
- **`dynalang/configs.yaml`** — All config presets. `defaults` is the base; environment-specific presets (`homegrid`, `vln`, `messenger`, etc.) and size presets (`small`, `medium`, `large`, `xlarge`) override specific values.
- Configs are composed: `--configs defaults homegrid small` merges in order.
- Regex-style keys (e.g., `.*\.layers: 2`) apply to all matching config paths.

### Logging
Supports terminal, Comet ML (`--loggers terminal,comet`), and Weights & Biases (`--use_wandb True`). Metrics go to `metrics.jsonl` and `scores.jsonl` in the logdir.
