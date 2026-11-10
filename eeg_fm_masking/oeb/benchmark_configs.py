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


_REBUTTAL_RUNS = [
    MaskRun(radius=r, length=l, framework=fw)
    for fw in ("mae", "jepa_noreg")
    for r, l in ((0.09, 2), (0.0, 1), (float("inf"), 2))
]


def get_configs_adaptation(datasets: list[str] | None = None) -> list[tuple[Experiment, dict]]:
    """Representative masks × (ridge_probe, lora, full_finetune) × 5 seeds via HF Hub."""
    import itertools
    from open_eeg_bench.default_configs.experiments import make_all_experiments
    from eeg_fm_masking.oeb.utils import hf_backbone

    base = make_all_experiments(
        datasets=datasets,
        heads=["linear_head"],
        finetuning_strategies=["ridge_probe", "lora", "full_finetune"],
        n_seeds=5,
    )
    bbs = {run: hf_backbone(run) for run in _REBUTTAL_RUNS}
    eams: list[tuple[Experiment, dict]] = []
    for run, exp in itertools.product(_REBUTTAL_RUNS, base):
        tr = {"device": "cuda"}
        if hasattr(exp.training, "lr"):
            tr["lr"] = 2e-4 if exp.finetuning.kind == "full" else 5e-4
        overrides = {
            "backbone": bbs[run],
            "dataset": {"preload": True},
            "training": tr,
            "infra": {"folder": EXCA_CACHE, "mode": "cached"},
        }
        md = {"run_id": MASK_SWEEP_RUN_IDS[run], "framework": run.framework, "radius": run.radius, "length": run.length}
        eams.append((exp.infra.clone_obj(overrides), md))
    return eams


def run_low_label_probe(
    datasets: list[str] = ("bcic2a", "arithmetic_zyma2019", "mdd_mumtaz2016", "bcic2020_3", "physionet"),
    fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    n_seeds: int = 5,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate representative HF checkpoints across training label fractions f in (0.10, 0.25, 0.50, 1.00)."""
    import numpy as np
    import torch
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
    from torch.utils.data import DataLoader, TensorDataset
    from open_eeg_bench.default_configs import ALL_DATASETS
    from open_eeg_bench.head import FlattenHead
    from open_eeg_bench.ridge_probe import StreamingRidgeProbeLearner, _make_projection_matrix
    from eeg_fm_masking.oeb.utils import hf_backbone

    rows = []
    for run in _REBUTTAL_RUNS:
        bb = hf_backbone(run)
        for ds_name in datasets:
            ds_cfg = ALL_DATASETS[ds_name]()
            _, tr_set, val_set, te_set, info = ds_cfg.setup(normalization=bb.normalization)
            model = bb.build(info["n_chans"], info["n_times"], info["n_outputs"], info["sfreq"], info["chs_info"]).to(device).eval()
            FlattenHead().apply(model, info["n_outputs"], bb.head_module_name)
            H, Y, P = {k: {s: [] for s in range(n_seeds)} for k in ("tr", "val", "te")}, {k: [] for k in ("tr", "val", "te")}, None
            with torch.no_grad():
                for k, ds in (("tr", tr_set), ("val", val_set), ("te", te_set)):
                    for b in DataLoader(ds, batch_size=64):
                        h = model(b[0].to(device)).cpu().to(torch.float64)
                        if P is None and h.shape[1] > 5000:
                            P = {s: _make_projection_matrix(h.shape[1], 5000, seed=s, device="cpu", dtype=torch.float64) for s in range(n_seeds)}
                        for s in range(n_seeds):
                            H[k][s].append(h @ P[s].T if P is not None else h)
                        Y[k].append(torch.as_tensor(b[1]))
            H = {k: {s: torch.cat(v[s]) for s in range(n_seeds)} for k, v in H.items()}
            Y = {k: torch.cat(v) for k, v in Y.items()}
            y_tr, y_te = Y["tr"].numpy(), Y["te"].numpy()
            for f in fractions:
                for s in range(n_seeds):
                    if f >= 1.0:
                        idx = np.arange(len(y_tr))
                    else:
                        sp = StratifiedShuffleSplit(1, train_size=f, random_state=s) if ds_cfg.n_classes else ShuffleSplit(1, train_size=f, random_state=s)
                        idx, _ = next(sp.split(np.zeros(len(y_tr)), y_tr))
                    lrn = StreamingRidgeProbeLearner(
                        torch.nn.Identity(), ds_cfg.n_classes, 256, 0, "cpu", None,
                        TensorDataset(H["val"][s], Y["val"]), None, s, "balanced", "float64", 0
                    )
                    lrn.fit(TensorDataset(H["tr"][s][idx], Y["tr"][idx]))
                    pred = lrn.predict(TensorDataset(H["te"][s], Y["te"]))
                    sc = r2_score(y_te, pred.ravel()) if ds_cfg.n_classes is None else balanced_accuracy_score(y_te, pred)
                    rows.append({"run_id": MASK_SWEEP_RUN_IDS[run], "framework": run.framework, "radius": run.radius, "length": run.length, "dataset": ds_name, "fraction": f, "seed": s, "score": float(sc)})
    return pd.DataFrame(rows)


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
    "adaptation": get_configs_adaptation,
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
