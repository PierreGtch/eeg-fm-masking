"""Hardcoded experiment configurations for OEB benchmarks.

Each ``get_configs_*`` function returns a list of ``(Experiment, metadata)``
tuples ready to be queried in exca read-only mode to re-collect cached
benchmark results.

To avoid re-downloading the actual ``.ckpt`` files (~250 MB each, not needed
in read-only mode), this module monkey-patches
``eeg_fm_masking.oeb.utils.download_checkpoint`` with a metadata-only variant.
It still queries the wandb run config (needed to reconstruct ``model_kwargs``,
which enters the exca UID) but skips both the checkpoint download and the
artifact metadata query — the symlink path is deterministic from
``(run_id, version)``, and ``epoch`` / ``step`` are not part of the UID.
"""

from pathlib import Path

import pandas as pd
from open_eeg_bench.experiment import Experiment

from eeg_fm_masking.oeb import utils as _utils
from eeg_fm_masking.oeb.constants import MASK_SWEEP_RUN_IDS, MaskRun
from eeg_fm_masking.oeb.utils import (
    CHECKPOINTS_DIR,
    EXCA_CACHE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_experiments_and_metadata,
)

_RUN_IDS: list[str] = list(MASK_SWEEP_RUN_IDS.values())  # 58 wandb run ids

_RUN_CONFIG_CACHE: dict[str, dict] = {}


def _metadata_only_download_checkpoint(
    run_id: str, version: str, scaler_override: str | None = None
):
    """Drop-in replacement for ``download_checkpoint`` that does no I/O on the
    .ckpt file and no artifact metadata query.

    Reconstructs the symlink path deterministically and pulls only the run
    config (cached in-process) — sufficient for matching the original exca
    UID in read-only mode.
    """
    # pylint: disable=import-outside-toplevel
    if run_id not in _RUN_CONFIG_CACHE:
        import wandb

        api = wandb.Api()
        _RUN_CONFIG_CACHE[run_id] = api.run(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}"
        ).config

    cfg = _RUN_CONFIG_CACHE[run_id]
    model_config = dict(cfg["framework"]["model"])
    dw = cfg["datamodule"]["dataset_wrapper"]
    model_config["scaler"] = (
        scaler_override if scaler_override is not None else dw["scaler"]
    )
    model_config["factor"] = dw["factor"]
    if dw.get("clip_sigma") is not None:
        model_config["clip_sigma"] = dw["clip_sigma"]

    # The file inside the wandb artifact is always ``model.ckpt``, so
    # ``download_checkpoint`` creates ``model_{run_id}-{version}.ckpt``.
    link = CHECKPOINTS_DIR / f"model_{run_id}-{version}.ckpt"
    return str(link), model_config, {}


_utils.download_checkpoint = _metadata_only_download_checkpoint


def get_configs_v9() -> list[tuple[Experiment, dict]]:
    """v9 of every mask-sweep checkpoint × 5 seeds × default datasets."""
    eams = build_experiments_and_metadata(
        run_ids=_RUN_IDS,
        versions=["v9"],
        n_seeds=5,
        mode="read-only",
    )
    expected = 58 * 12 * 5  # 58 runs * 12 datasets * 5 seeds
    assert len(eams) == expected, f"{len(eams)} vs {expected}"
    return eams


def get_configs_timecourse() -> list[tuple[Experiment, dict]]:
    """v0..v9 of every mask-sweep checkpoint × 3 seed × default datasets."""
    versions = [f"v{i}" for i in range(10)]
    eams = build_experiments_and_metadata(
        run_ids=_RUN_IDS,
        versions=versions,
        n_seeds=3,
        mode="read-only",
    )
    expected = 58 * 10 * 12 * 3  # 58 runs * 10 versions * 12 datasets * 3 seeds
    assert len(eams) == expected, f"{len(eams)} vs {expected}"
    return eams


def get_configs_timecourse_best() -> list[tuple[Experiment, dict]]:
    """v0..v9 of the best (r=9, L=2) config × 5 seed × default datasets."""
    versions = [f"v{i}" for i in range(10)]
    run_ids = [
        MASK_SWEEP_RUN_IDS[MaskRun(framework=framework, radius=0.09, length=2)]
        for framework in ["mae", "jepa_noreg"]
    ]
    eams = build_experiments_and_metadata(
        run_ids=run_ids,
        versions=versions,
        n_seeds=5,
        mode="read-only",
    )
    expected = 2 * 10 * 12 * 5  # 2 runs * 10 versions * 12 datasets * 5 seeds
    assert len(eams) == expected, f"{len(eams)} vs {expected}"
    return eams


def get_configs_reve_baseline() -> list[tuple[Experiment, dict]]:
    """Pretrained REVE backbone × 5 seeds × default datasets (no run_ids)."""
    # pylint: disable=import-outside-toplevel
    from open_eeg_bench.default_configs.backbones import reve
    from open_eeg_bench.default_configs.experiments import make_all_experiments

    base_experiments = make_all_experiments(
        heads=["linear_head"],
        finetuning_strategies=["ridge_probe"],
        n_seeds=5,
    )
    overrides = {
        "backbone": reve(),
        "dataset": {"preload": True},
        "training": {"device": "cuda"},
        "infra": {"folder": EXCA_CACHE, "mode": "read-only"},
    }
    eams: list[tuple[Experiment, dict]] = [
        (exp.infra.clone_obj(overrides), {}) for exp in base_experiments
    ]
    expected = 12 * 5  # 12 datasets * 5 seeds
    assert len(eams) == expected, f"{len(eams)} vs {expected}"
    return eams


def collect_results(
    eams: list[tuple[Experiment, dict]], out: str | Path
) -> pd.DataFrame:
    rows = []
    for exp, metadata in eams:
        row = {
            **metadata,
            "dataset": exp.dataset.hf_id,
            "finetuning": exp.finetuning.kind,
            "head": exp.head.kind,
            "seed": exp.seed,
            "training": exp.training.kind,
            "scaler": exp.backbone.model_kwargs.get("scaler"),
            "status": exp.infra.status(),
        }
        if row["status"] == "completed":
            row.update(exp.infra.job().result())
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


CONFIGS_DICT = {
    "v9": get_configs_v9,
    "timecourse": get_configs_timecourse,
    "timecourse_best": get_configs_timecourse_best,
    "reve_baseline": get_configs_reve_baseline,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect OEB benchmark results.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        choices=CONFIGS_DICT.keys(),
        nargs="+",
        default=list(CONFIGS_DICT.keys()),
        help="Which benchmark experiment to collect results for.",
    )
    args = parser.parse_args()

    for config_name in args.experiment_name:
        out = f"oeb_results_{config_name}.csv"
        get_configs_fn = CONFIGS_DICT[config_name]
        print(f"Collecting results for {get_configs_fn.__name__}...")
        eams = get_configs_fn()
        collect_results(eams, out)
