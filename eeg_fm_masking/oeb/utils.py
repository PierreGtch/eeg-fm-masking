from __future__ import annotations

import contextlib
import itertools
import os
import re
import time
from pathlib import Path

from open_eeg_bench.experiment import Experiment

# All paths default to local user-cache directories. On a cluster with a
# shared filesystem, point these env vars at a project-space directory so
# that exca caches, downloaded HuggingFace datasets, wandb artifacts and
# checkpoints survive across jobs.
_DEFAULT_CACHE_ROOT = Path("~/.cache/eeg_fm_masking").expanduser()
EXCA_CACHE = os.environ.get("OEB_EXCA_CACHE", str(_DEFAULT_CACHE_ROOT / "exca"))
CHECKPOINTS_DIR = Path(
    os.environ.get("OEB_CHECKPOINTS_DIR", str(_DEFAULT_CACHE_ROOT / "checkpoints"))
)
HF_HOME = os.environ.get("HF_HOME", str(_DEFAULT_CACHE_ROOT / "hf_cache"))
WANDB_CACHE_DIR = os.environ.get(
    "WANDB_CACHE_DIR", str(_DEFAULT_CACHE_ROOT / "wandb_cache/cache")
)
WANDB_ARTIFACT_DIR = os.environ.get(
    "WANDB_ARTIFACT_DIR", str(_DEFAULT_CACHE_ROOT / "wandb_cache/artifacts")
)
# Wandb workspace (entity/project) holding the pretraining runs. Must be
# overridden via env vars to point at the actual workspace before launching.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "eeg-fm-masking")


def setup_env() -> None:
    """Set the env vars wandb / huggingface read at import time. Must run
    before any wandb/hf/torch import."""
    os.environ.setdefault("HF_HOME", HF_HOME)
    # Avoid distributed-filesystem fcntl.flock contention by skipping the
    # MNE config file read on launch (concurrent jobs all hit the same file).
    os.environ.setdefault("MNE_LOGGING_LEVEL", "INFO")
    # Shared mutex dir for cross-node wandb.init serialization (see _cross_node_lock).
    os.environ.setdefault("WANDB_LOCK_DIR", EXCA_CACHE)
    os.environ.setdefault("WANDB_CACHE_DIR", WANDB_CACHE_DIR)
    os.environ.setdefault("WANDB_ARTIFACT_DIR", WANDB_ARTIFACT_DIR)


@contextlib.contextmanager
def _cross_node_lock(lock_path: Path, timeout: float = 900, poll: float = 3):
    """Atomic cross-node mutex via O_CREAT|O_EXCL on a shared filesystem.

    Multiple SLURM workers on different nodes may try to resume the same
    wandb run concurrently, which silently drops writes from all-but-one
    worker. flock is not always reliable on distributed filesystems, so we
    use atomic file creation instead. Stale locks older than ``timeout``
    are reclaimed.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()}@{os.uname().nodename}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > timeout:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise RuntimeError(f"Timeout waiting for lock {lock_path}")
            time.sleep(poll)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def run_and_log_meta_experiment(
    experiments_and_metadata: list[
        tuple[Experiment, dict[str, int | float | str | None]]
    ],
    max_experiments_running_per_node: int,
) -> None:
    """Run a list of experiments as a MetaExperiment and log results to wandb."""
    # lazy imports such that we can instantiate this function from a login node:
    # pylint: disable=import-outside-toplevel
    import pandas as pd
    import wandb
    from exca import TaskInfra

    from open_eeg_bench.helpers import MetaExperiment
    from open_eeg_bench.experiment import collect_completed_results

    experiments, metadata = zip(*experiments_and_metadata)

    assert all({"run_id"} <= set(md.keys()) for md in metadata)

    experiments = list(experiments)
    meta_exp = MetaExperiment(
        experiments=experiments,
        n_jobs=max_experiments_running_per_node,
        infra=TaskInfra(),  # run locally, blocking, no cluster or cache
    )
    meta_exp.run()

    results = collect_completed_results(experiments, wait=False, collect_all=True)

    metadata_df = pd.DataFrame(metadata)
    assert len(results) == len(metadata_df)
    results = pd.concat([results, metadata_df], axis=1)
    results_filtered = results[results["status"] == "completed"]

    n_fail = len(results) - len(results_filtered)
    if n_fail > 0:
        raise ValueError(
            f"{n_fail}/ {len(results)} experiments did not complete successfully."
        )


def download_checkpoint(run_id: str, version: str, scaler_override: str | None = None):
    """Download a wandb model artifact, symlink it locally, return path/config/step."""
    import wandb

    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    model_config = run.config["framework"]["model"]
    dw = run.config["datamodule"]["dataset_wrapper"]
    model_config["scaler"] = (
        scaler_override if scaler_override is not None else dw["scaler"]
    )
    model_config["factor"] = dw["factor"]
    # Only add clip_sigma when set, to keep cache UIDs stable for old runs.
    if dw.get("clip_sigma") is not None:
        model_config["clip_sigma"] = dw["clip_sigma"]

    artifact = api.artifact(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/model-{run_id}:{version}", type="model"
    )
    ckpt = next(Path(artifact.download()).glob("*.ckpt"), None)
    if ckpt is None:
        raise FileNotFoundError(f"No .ckpt file in artifact for {run_id}:{version}")
    ckpt = ckpt.resolve()

    # Suffix the symlink with run_id+version so different runs/versions don't
    # collide on downstream cache keys (e.g. "epoch=10-step=1000.ckpt").
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    link = CHECKPOINTS_DIR / f"{ckpt.stem}_{run_id}-{version}{ckpt.suffix}"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(ckpt)

    fname = artifact.metadata["original_filename"]
    m = re.fullmatch(r"epoch=(\d+)-step=(\d+)\.ckpt", fname)
    if m is None:
        raise ValueError(f"Unexpected checkpoint filename: {fname!r}")
    return str(link), model_config, {"epoch": int(m.group(1)), "step": int(m.group(2))}


def resolve_run_version_pairs(
    run_ids: list[str], versions: list[str]
) -> list[tuple[str, str]]:
    """Expand --run_ids/--versions into (run_id, version) pairs."""
    if "all" in versions and len(versions) > 1:
        raise ValueError("'all' can not be combined with other versions.")

    api = None
    out = []
    for run_id in run_ids:
        for version in versions:
            if version == "all":
                if api is None:
                    import wandb

                    api = wandb.Api()
                artifacts = api.artifact_versions(
                    type_name="model",
                    name=f"{WANDB_ENTITY}/{WANDB_PROJECT}/model-{run_id}",
                )
                run_versions = sorted(
                    {a.version for a in artifacts},
                    key=lambda v: (
                        int(v[1:]) if v.startswith("v") and v[1:].isdigit() else -1
                    ),
                )
                if not run_versions:
                    raise ValueError(f"No artifact versions found for {run_id}")
                out.extend([(run_id, v) for v in run_versions])
            else:
                out.append((run_id, version))
    return out


def build_experiments_and_metadata(
    run_ids: list[str],
    versions: list[str],
    heads: list[str] = ("linear_head",),
    datasets: list[str] | None = None,
    finetuning: list[str] = ("ridge_probe",),
    n_seeds: int = 1,
    mode: str = "cached",
    scaler_override: str | None = None,
) -> list[tuple[Experiment, dict]]:
    """Build every (experiment, metadata) tuple for the given run_ids × versions × base experiments.

    Experiments are cloned with the submission-time overrides (backbone pointing
    at the downloaded checkpoint, NFS exca cache, etc.). Pass ``mode="read-only"``
    to iterate cached results without risking re-runs.
    """
    # pylint: disable=import-outside-toplevel
    from open_eeg_bench.backbone import PretrainedBackbone
    from open_eeg_bench.default_configs.experiments import make_all_experiments

    pairs = resolve_run_version_pairs(list(run_ids), list(versions))
    base_experiments = make_all_experiments(
        heads=list(heads),
        datasets=datasets,
        finetuning_strategies=list(finetuning),
        n_seeds=n_seeds,
    )

    # Pre-download checkpoints once per (run_id, version) pair so the sweep
    # below doesn't trigger N redundant artifact downloads.
    ckpts = {
        pair: download_checkpoint(*pair, scaler_override=scaler_override)
        for pair in pairs
    }

    experiments_and_metadata: list[tuple[Experiment, dict]] = []
    for (run_id, version), exp in itertools.product(pairs, base_experiments):
        ckpt_path, model_cfg, ckpt_meta = ckpts[(run_id, version)]
        overrides = {
            "backbone": PretrainedBackbone(
                model_cls="eeg_fm_masking.oeb.wrapper.ContextualEncoderBenchmarkWrapper",
                checkpoint_path=ckpt_path,
                model_kwargs=dict(model_cfg),
            ),
            "dataset": {"preload": True},
            "training": {"device": "cuda"},
            "infra": {"folder": EXCA_CACHE, "mode": mode},
        }
        metadata = {
            **ckpt_meta,
            "run_id": run_id,
            "version": version,
        }
        experiments_and_metadata.append((exp.infra.clone_obj(overrides), metadata))
    return experiments_and_metadata
