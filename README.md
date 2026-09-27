
# What masking geometry works best for EEG foundation models?

*A controlled evaluation across MAE and JEPA.*

[![Website](https://img.shields.io/badge/%F0%9F%8C%90%20Website-paper%20explained-blue)](https://pierregtch.github.io/eeg-fm-masking)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/PierreGtch/eeg-fm-masking)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenEEGBench](https://img.shields.io/badge/%F0%9F%8F%86%20evaluated%20on-OpenEEGBench-purple)](https://github.com/braindecode/OpenEEGBench)
[![GitHub stars](https://img.shields.io/github/stars/PierreGtch/eeg-fm-masking?style=flat&logo=github)](https://github.com/PierreGtch/eeg-fm-masking/stargazers)

This repository contains the pre-training and downstream-evaluation code used
to produce the results presented in the article. 

## Repository structure

```
.
├── mask_sweep_full.py          — pre-training entry point (mask-geometry sweep)
├── regenerate_reve_cache.py    — build the pre-shuffled shard cache from REVE
├── eeg_fm_masking/               — Python package
│   ├── configs/                — pydantic configs for trainer / framework / SLURM
│   ├── reve_cache/             — pre-shuffled shard cache reader + writer
│   ├── modules.py, models.py,  — model architecture (ContextualEncoder + MaskMaker)
│   │   transformer.py, ...
│   ├── pl_mae.py, pl_ssl.py    — Lightning modules for MAE and JEPA
│   └── oeb/                    — downstream evaluation on OpenEEGBench
│       ├── benchmark_predefined.py     — launch one of the predefined OEB experiments
│       ├── benchmark_reve_baseline.py  — launch the REVE-paper baseline on OEB
│       ├── benchmark_configs.py        — predefined experiment configs + result collector
│       └── wrapper.py                  — adapts our encoder to OEB's model interface
├── results/                    — cached CSVs of every reported experiment
└── pyproject.toml
```

## Reproducing the paper

### 1. Install

```bash
pip install -e .
```

Pre-training and OEB evaluation both run on SLURM via [exca](https://github.com/facebookresearch/exca)
+ submitit. Cluster-specific values (partition, account, QOS, cache paths,
wandb workspace) are read from environment variables — see the docstrings in
the launcher scripts.

### 2. Pre-training (mask-geometry sweep)

Pre-training reads from a *pre-shuffled shard cache* derived from the
[REVE dataset](https://huggingface.co/datasets/brain-bzh/reve-dataset).
Windows are routed to shards at generation time so a single sequential read
per shard already yields a fully diverse mini-batch.

After downloading REVE locally, generate the cache once with:

```bash
python regenerate_reve_cache.py \
    --split train \
    --reve_dir /path/to/reve-dataset \
    --out_dir /path/to/reve_shuffled/train \
    --n_workers 32
```

Then submit the sweep:

```bash
export REVE_TRAIN_CACHE_DIR=/path/to/reve_shuffled/train
export REVE_POSITIONS_DIR=/path/to/reve-dataset/positions
export WANDB_ENTITY=<your-wandb-entity>

python mask_sweep_full.py
```

This submits 58 SLURM jobs (2 frameworks × 29 mask configurations, 10 epochs
each) as a single `infra.job_array()` submission.

### 3. Downstream evaluation on OpenEEGBench

```bash
# Submit (e.g. the v9 experiment: 58 mask runs × 12 datasets × 5 seeds).
python -m eeg_fm_masking.oeb.benchmark_predefined \
    --experiment_name v9 \
    --slurm_partition <PARTITION> \
    --slurm_account <ACCOUNT>

# Once jobs are done, collect cached results into results/oeb_results_v9.csv
python -m eeg_fm_masking.oeb.benchmark_configs --experiment_name v9
```

The four predefined experiments (`v9`, `timecourse`, `timecourse_best`,
`reve_baseline`) are declared in
[`eeg_fm_masking/oeb/benchmark_configs.py`](eeg_fm_masking/oeb/benchmark_configs.py).
The pre-trained-checkpoint run IDs are listed in
[`eeg_fm_masking/oeb/constants.py`](eeg_fm_masking/oeb/constants.py).

## Cached results

Every CSV under [`results/`](results/) is the exact output used in the paper
and can be re-collected from the exca cache without re-running anything:

| File | Contents |
|------|----------|
| `oeb_results_v9.csv` | 58 mask configs × 12 datasets × 5 seeds (final epoch) |
| `oeb_results_timecourse.csv` | 58 mask configs × 10 epochs × 12 datasets × 3 seeds |
| `oeb_results_timecourse_best.csv` | best mask config × 10 epochs × 12 datasets × 5 seeds |
| `oeb_results_reve_baseline.csv` | REVE backbone baseline × 12 datasets × 5 seeds |
| `oeb_results_non-fm_baselines.csv` | non-foundation-model baselines |
