"""Benchmark the REVE baseline backbone on OpenEEGBench (SLURM)."""

import argparse
import contextlib
import os
from pathlib import Path

from eeg_fm_masking.oeb.utils import EXCA_CACHE, setup_env

setup_env()

# Optional: directory holding the eeg_fm_masking source on the cluster, prepended
# to PYTHONPATH so SLURM jobs can import it. Leave the env var unset if the
# package is already installed in the job's Python environment.
_PKG_SRC_DIR = os.environ.get("EEG_FM_MASKING_DIR", "")
if _PKG_SRC_DIR:
    os.environ["PYTHONPATH"] = ":".join(
        p for p in (_PKG_SRC_DIR, os.environ.get("PYTHONPATH", "")) if p
    )

# Patch submitit's clean_env to preserve SLURM_CONF inside spawned jobs.
import submitit.helpers

_orig_clean_env = submitit.helpers.clean_env


@contextlib.contextmanager
def _clean_env_preserve(*args, **kwargs):
    slurm_conf = os.environ.get("SLURM_CONF")
    with _orig_clean_env(*args, **kwargs):
        if slurm_conf is not None:
            os.environ["SLURM_CONF"] = slurm_conf
        yield


submitit.helpers.clean_env = _clean_env_preserve

from open_eeg_bench.default_configs.experiments import make_all_experiments
from open_eeg_bench.default_configs.backbones import reve
from open_eeg_bench.helpers import MetaExperiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--finetuning", nargs="*", default=["ridge_probe"])
    p.add_argument("--heads", nargs="*", default=["linear_head"])
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument(
        "--max_experiments_running_per_node",
        type=int,
        default=5,
        help=(
            "joblib parallelism within a single SLURM job. Lower this "
            "(e.g. 1) if experiments OOM when run in parallel — preloading "
            "large datasets into memory multiplies RAM usage by this factor."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["cached", "retry", "force", "read-only"],
        default="cached",
        help="exca mode for the inner experiments. 'retry' re-runs only failures.",
    )
    p.add_argument("--timeout_min", type=int, default=120)
    p.add_argument(
        "--slurm_partition",
        type=str,
        default=os.environ.get("SLURM_PARTITION", ""),
        help="SLURM partition name (default: $SLURM_PARTITION env var).",
    )
    p.add_argument(
        "--slurm_account",
        type=str,
        default=os.environ.get("SLURM_ACCOUNT", ""),
        help="SLURM account (default: $SLURM_ACCOUNT env var).",
    )
    p.add_argument(
        "--slurm_qos",
        type=str,
        default=os.environ.get("SLURM_QOS", ""),
        help="SLURM QOS (default: $SLURM_QOS env var).",
    )
    p.add_argument("--mem_gb", type=int, default=96)
    p.add_argument("--cpus_per_task", type=int, default=10)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--collect", action="store_true")
    return p.parse_args()


def _force_resubmit_meta_jobs(meta_experiments: list) -> None:
    """Workaround for an exca ``job_array`` bug.

    When the leader infra (``meta_experiments[0].infra.job_array()``) is itself
    one of the array members, the inner loop sets ``state.computed=True`` on
    every infra including the leader. That downgrades the leader's
    ``_effective_mode`` from ``"force"`` to ``"cached"``, so the ``to_clear``
    branch is skipped and stale failed/cancelled meta jobs are never resubmitted.

    We pre-clear them via the public ``clear_job()`` API so they appear as
    ``"not submitted"`` once we enter ``job_array()``.
    """
    for me in meta_experiments:
        if me.infra.status() != "not submitted":
            me.infra.clear_job()


def main() -> None:
    args = parse_args()

    base_experiments = make_all_experiments(
        heads=args.heads,
        datasets=args.datasets,
        finetuning_strategies=args.finetuning,
        n_seeds=args.n_seeds,
    )

    overrides = {
        "backbone": reve(),
        "dataset": {"preload": True},
        "training": {"device": "cuda"},
        "infra": {"folder": EXCA_CACHE, "mode": args.mode},
    }
    experiments = [exp.infra.clone_obj(overrides) for exp in base_experiments]

    if not args.collect:
        if args.mode != "force":
            n_experiments = len(experiments)
            # remove experiments that are already completed to avoid redundant benchmarking
            experiments = [
                exp for exp in experiments if exp.infra.status() != "completed"
            ]
            n_filtered_experiments = len(experiments)
            if n_filtered_experiments != n_experiments:
                print(
                    f"Filtered out {n_experiments - n_filtered_experiments} already completed experiments, "
                    f"{n_filtered_experiments} remaining."
                )
            if n_filtered_experiments == 0:
                print("No experiments to run. Exiting.")
                return

        slurm_extra: dict = {"gpus": args.gpus}
        if args.slurm_qos:
            slurm_extra["qos"] = args.slurm_qos
        meta_infra = {
            "folder": EXCA_CACHE,
            "cluster": "slurm",
            "mode": "force",
            "slurm_partition": args.slurm_partition,
            "slurm_account": args.slurm_account,
            "timeout_min": args.timeout_min,
            "nodes": 1,
            "mem_gb": args.mem_gb,
            "cpus_per_task": args.cpus_per_task,
            "slurm_additional_parameters": slurm_extra,
        }
        groups = [experiments[i :: args.max_workers] for i in range(args.max_workers)]
        meta_experiments = [
            MetaExperiment(
                experiments=group,
                n_jobs=args.max_experiments_running_per_node,
                infra=meta_infra,
            )
            for group in groups
        ]
        print(
            f"Prepared {len(meta_experiments)} meta-experiments "
            f"containing each {[len(g) for g in groups]}."
        )

        _force_resubmit_meta_jobs(meta_experiments)

        with meta_experiments[0].infra.job_array() as array:
            array.extend(meta_experiments)
    else:
        import pandas as pd

        experiments = [
            exp.infra.clone_obj({"infra": {"mode": "read-only"}}) for exp in experiments
        ]
        results = []
        for exp in experiments:
            row = {
                "backbone": exp.backbone.model_cls,
                "dataset": exp.dataset.hf_id,
                "finetuning": exp.finetuning.kind,
                "head": exp.head.kind,
                "seed": exp.seed,
                "training": exp.training.kind,
                "status": exp.infra.status(),
            }
            if row["status"] == "completed":
                job = exp.infra.job()
                row.update(job.result())
            results.append(row)
        df = pd.DataFrame(results)
        print(df)
        out_path = Path("reve_baseline_oeb_results.csv").resolve()
        df.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
