"""ICLR 2027 Rebuttal Analyses: LODO/L2DO stability, compute attribution, realized-ratio slice, and HF collapse diagnostics."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from eeg_fm_masking.configs.architectures import ContextualEncoderConfig
from eeg_fm_masking.modules import CovLoss, VarLoss
from eeg_fm_masking.oeb.constants import MASK_SWEEP_RUN_IDS

RESULTS = Path(__file__).resolve().parent / "results"
RUN_META = {
    rid: (
        r.framework,
        "one" if r.radius == 0 else ("all" if math.isinf(r.radius) else f"{int(round(r.radius * 100))}cm"),
        r.length,
    )
    for r, rid in MASK_SWEEP_RUN_IDS.items()
}
REALIZED_RHO = {
    ("one", 1): 55.0, ("one", 2): 55.8, ("one", 4): 55.7, ("one", 8): 54.4, ("one", 16): 51.8, ("one", 33): 55.4,
    ("6cm", 1): 56.2, ("6cm", 2): 56.0, ("6cm", 4): 55.2, ("6cm", 8): 52.8, ("6cm", 16): 50.8, ("6cm", 33): 55.4,
    ("9cm", 1): 53.9, ("9cm", 2): 53.4, ("9cm", 4): 52.0, ("9cm", 8): 52.1, ("9cm", 16): 51.9, ("9cm", 33): 55.7,
    ("12cm", 1): 54.5, ("12cm", 2): 53.8, ("12cm", 4): 55.5, ("12cm", 8): 54.8, ("12cm", 16): 57.5, ("12cm", 33): 70.0,
    ("all", 1): 55.4, ("all", 2): 55.4, ("all", 4): 58.2, ("all", 8): 62.6, ("all", 16): 66.6,
}


def load_v9() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(RESULTS / "oeb_results_v9.csv")
    df["score"] = df["test_balanced_accuracy"].fillna(df["test_r2"])
    df[["fw", "r", "L"]] = pd.DataFrame(df["run_id"].map(RUN_META).tolist(), index=df.index)
    ds_means = df.groupby(["dataset", "run_id"])["score"].mean()
    s_min, s_max = ds_means.groupby("dataset").min(), ds_means.groupby("dataset").max()
    df["norm"] = (df["score"] - df["dataset"].map(s_min)) / (df["dataset"].map(s_max) - df["dataset"].map(s_min))
    df["cell"] = df["r"] + "_L" + df["L"].astype(str)
    return df, s_min, s_max


def run_csv_analyses() -> dict:
    df, s_min, s_max = load_v9()
    datasets = sorted(df["dataset"].unique())
    lodo = {}
    for fw in ("mae", "jepa_noreg"):
        sub = df[df["fw"] == fw].copy()
        sub["seed_rk"] = sub.groupby(["dataset", "seed"])["score"].rank(ascending=False)
        piv_norm = sub.pivot_table(index="cell", columns="dataset", values="norm", aggfunc="mean")
        piv_rk = sub.pivot_table(index="cell", columns="dataset", values="seed_rk", aggfunc="mean")
        l1_top1 = sum(piv_norm.drop(columns=[d]).mean(axis=1).idxmax() == "9cm_L2" for d in datasets)
        l1_top3 = sum(piv_rk.drop(columns=[d]).mean(axis=1).rank().loc["9cm_L2"] <= 3 for d in datasets)
        l2_top3 = sum(piv_rk.drop(columns=list(p)).mean(axis=1).rank().loc["9cm_L2"] <= 3 for p in itertools.combinations(datasets, 2))
        lodo[fw] = {"lodo_norm_top1_of_12": int(l1_top1), "lodo_rank_top3_of_12": int(l1_top3), "l2do_rank_top3_of_66": int(l2_top3)}

    tc = pd.concat([pd.read_csv(RESULTS / "oeb_results_timecourse.csv"), pd.read_csv(RESULTS / "oeb_results_timecourse_best.csv"), pd.read_csv(RESULTS / "oeb_results_v9.csv")])
    tc = tc.drop_duplicates(subset=["run_id", "version", "dataset", "seed"]).copy()
    tc["score"] = tc["test_balanced_accuracy"].fillna(tc["test_r2"])
    tc[["fw", "r", "L"]] = pd.DataFrame(tc["run_id"].map(RUN_META).tolist(), index=tc.index)
    tc["norm"] = (tc["score"] - tc["dataset"].map(s_min)) / (tc["dataset"].map(s_max) - tc["dataset"].map(s_min))
    tc["epoch"] = tc["version"].str.slice(1).astype(int) + 1
    ep_curve = tc.groupby(["fw", "r", "L", "epoch"])["norm"].mean().unstack("epoch")

    agg = df.groupby(["fw", "r", "L"])["norm"].mean().reset_index()
    agg["rho_real"] = [REALIZED_RHO[(r, int(l))] for r, l in zip(agg["r"], agg["L"])]
    matched = agg[agg["rho_real"].between(53.0, 56.2)]

    base = pd.read_csv(RESULTS / "oeb_results_non-fm_baselines.csv")
    return {
        "lodo_l2do": lodo,
        "epoch_trajectories": {
            "mae_9cm_L2": ep_curve.loc[("mae", "9cm", 2)].round(4).to_dict(),
            "jepa_9cm_L2": ep_curve.loc[("jepa_noreg", "9cm", 2)].round(4).to_dict(),
        },
        "matched_rho_L2_spread": {
            fw: round(float(matched[(matched["fw"] == fw) & (matched["L"] == 2)]["norm"].max() - matched[(matched["fw"] == fw) & (matched["L"] == 2)]["norm"].min()), 4)
            for fw in ("mae", "jepa_noreg")
        },
        "fit_time_mean_s": {
            "ridge_probe": round(float(df.groupby(["run_id", "dataset"])["fit_time"].mean().mean()), 2),
            "supervised_sgd": round(float(base.groupby(["backbone", "dataset"])["fit_time"].mean().mean()), 2),
        },
    }


@torch.no_grad()
def run_hf_collapse_check(
    epochs: tuple[int, ...] = (1, 2, 3, 5, 7, 10),
    keys: tuple[str, ...] = ("jepa_rall_L2", "jepa_r9cm_L2", "mae_rall_L2", "mae_r9cm_L2"),
) -> dict:
    from open_eeg_bench.default_configs import ALL_DATASETS
    from eeg_fm_masking.functions import scale_signal
    from eeg_fm_masking.oeb.utils import hf_backbone
    from eeg_fm_masking.oeb.constants import MaskRun

    ds_cfg = ALL_DATASETS["bcic2a"](preload=True)
    bb = hf_backbone(MaskRun(0.09, 2, "jepa_noreg"))
    _, _, val_set, _, info = ds_cfg.setup(normalization=bb.normalization)
    X_raw = torch.stack([torch.as_tensor(val_set[i][0], dtype=torch.float32) for i in range(min(64, len(val_set)))])
    X = scale_signal(X_raw, 1e6, "median_std_clip", 15.0)
    pos = torch.tensor([ch["loc"][:3] for ch in info["chs_info"]], dtype=torch.float32).unsqueeze(0).expand(len(X), -1, -1).contiguous()

    vl, cl = VarLoss(1.0), CovLoss()
    out = {}
    for key in keys:
        repo = f"PierreGtch/eeg-fm-masking_{key}"
        raw = json.loads(Path(hf_hub_download(repo, "config.json", token=False)).read_text())
        cfg = ContextualEncoderConfig(**{k: v for k, v in raw.items() if k in ContextualEncoderConfig.model_fields})
        enc, fe = cfg.create_instance()
        mod = torch.nn.Module()
        mod.feature_encoder, mod.model = fe, enc
        out[key] = {}
        for ep in epochs:
            fname = "model.safetensors" if ep == 10 else f"epoch_{ep:02d}/model.safetensors"
            sd = load_file(hf_hub_download(repo, fname, token=False))
            mod.load_state_dict(sd, strict=False)
            mod.eval()
            z = mod.model({"X": X, "ch_pos": pos, "local_features": mod.feature_encoder(X)}, apply_mask=False)["ctx_features"].float()
            M = z.mean(0, keepdim=True)
            sig_pos = float(M.squeeze(0).reshape(-1, z.shape[-1]).std(0).mean())
            sig_eeg = float((z - M).std(0).mean())
            flat = einops.rearrange(z, "b c t d -> (b c t) d")
            per_dim_std = float(flat.std(dim=0).mean())
            s = torch.linalg.svdvals(flat - flat.mean(0, keepdim=True))
            p = s / s.sum().clamp(min=1e-12)
            rankme = float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))
            out[key][f"ep{ep:02d}"] = {
                "sigma_token": round(sig_pos, 4),
                "sigma_sample_at_ct": round(sig_eeg, 4),
                "signal_to_pos_ratio": round(sig_eeg / max(sig_pos, 1e-8), 4),
                "per_dim_std": round(per_dim_std, 4),
                "var_loss": round(float(vl(flat)), 4),
                "cov_loss": round(float(cl(flat)), 4),
                "rankme": round(rankme, 2),
            }
    return out


if __name__ == "__main__":
    summary = run_csv_analyses()
    summary["hf_collapse"] = run_hf_collapse_check()
    print(json.dumps(summary, indent=2))
