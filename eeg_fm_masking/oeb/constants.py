import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskRun:
    radius: float
    length: int
    framework: str


# ---------------------------------------------------------------------------
# (radius_m, length_patches, framework) -> wandb run_id for the 58 mask-sweep
# pre-training runs.
# ---------------------------------------------------------------------------
MASK_SWEEP_RUN_IDS: dict[MaskRun, str] = {
    MaskRun(radius=0.0,      length=1,  framework="mae"):        "hnhbuysb",
    MaskRun(radius=math.inf, length=2,  framework="mae"):        "0hfltxhc",
    MaskRun(radius=0.06,     length=33, framework="mae"):        "ab3zq84f",
    MaskRun(radius=0.06,     length=2,  framework="mae"):        "mi2jxz5g",
    MaskRun(radius=0.0,      length=1,  framework="jepa_noreg"): "se0wazkq",
    MaskRun(radius=math.inf, length=2,  framework="jepa_noreg"): "a8gdctp4",
    MaskRun(radius=0.06,     length=33, framework="jepa_noreg"): "4jsjobya",
    MaskRun(radius=0.06,     length=2,  framework="jepa_noreg"): "6ikegvah",

    MaskRun(radius=0.0,      length=2,  framework="mae"):        "qizardwg",
    MaskRun(radius=0.0,      length=4,  framework="mae"):        "8hesnn4j",
    MaskRun(radius=0.0,      length=8,  framework="mae"):        "h4ckwdd0",
    MaskRun(radius=0.0,      length=16, framework="mae"):        "46gmjq13",
    MaskRun(radius=0.0,      length=33, framework="mae"):        "wl8bkq9x",
    MaskRun(radius=0.06,     length=1,  framework="mae"):        "du0xq3k8",
    MaskRun(radius=0.06,     length=4,  framework="mae"):        "d2k3me28",
    MaskRun(radius=0.06,     length=8,  framework="mae"):        "5jxbqr35",
    MaskRun(radius=0.06,     length=16, framework="mae"):        "vlt5ckzw",
    MaskRun(radius=0.09,     length=1,  framework="mae"):        "yncl6get",
    MaskRun(radius=0.09,     length=2,  framework="mae"):        "ado2flzb",
    MaskRun(radius=0.09,     length=4,  framework="mae"):        "v78z6b50",
    MaskRun(radius=0.09,     length=8,  framework="mae"):        "z8paagta",
    MaskRun(radius=0.09,     length=16, framework="mae"):        "wbwysdic",
    MaskRun(radius=0.09,     length=33, framework="mae"):        "kqa3qhu7",
    MaskRun(radius=0.12,     length=1,  framework="mae"):        "0e8r79dk",
    MaskRun(radius=0.12,     length=2,  framework="mae"):        "eeidax1z",
    MaskRun(radius=0.12,     length=4,  framework="mae"):        "4stxb7s2",
    MaskRun(radius=0.12,     length=8,  framework="mae"):        "9p8xmw0q",
    MaskRun(radius=0.12,     length=16, framework="mae"):        "nd67puc3",
    MaskRun(radius=0.12,     length=33, framework="mae"):        "7aipvvk5",
    MaskRun(radius=math.inf, length=1,  framework="mae"):        "yqzaq5j1",
    MaskRun(radius=math.inf, length=4,  framework="mae"):        "ds29fgcl",
    MaskRun(radius=math.inf, length=8,  framework="mae"):        "c9pyckaw",
    MaskRun(radius=math.inf, length=16, framework="mae"):        "q7ui5yvk",

    MaskRun(radius=0.0,      length=2,  framework="jepa_noreg"): "fb8blr82",
    MaskRun(radius=0.0,      length=4,  framework="jepa_noreg"): "km089re9",
    MaskRun(radius=0.0,      length=8,  framework="jepa_noreg"): "394mrb8x",
    MaskRun(radius=0.0,      length=16, framework="jepa_noreg"): "e76drwsj",
    MaskRun(radius=0.0,      length=33, framework="jepa_noreg"): "x51ltw3v",
    MaskRun(radius=0.06,     length=1,  framework="jepa_noreg"): "p42catef",
    MaskRun(radius=0.06,     length=4,  framework="jepa_noreg"): "fyxl8ibt",
    MaskRun(radius=0.06,     length=8,  framework="jepa_noreg"): "vla22qxd",
    MaskRun(radius=0.06,     length=16, framework="jepa_noreg"): "r1kj23po",
    MaskRun(radius=0.09,     length=1,  framework="jepa_noreg"): "biy1de7c",
    MaskRun(radius=0.09,     length=2,  framework="jepa_noreg"): "tl2ef402",
    MaskRun(radius=0.09,     length=4,  framework="jepa_noreg"): "sxgumypr",
    MaskRun(radius=0.09,     length=8,  framework="jepa_noreg"): "l0un9a30",
    MaskRun(radius=0.09,     length=16, framework="jepa_noreg"): "8e9ucmvt",
    MaskRun(radius=0.09,     length=33, framework="jepa_noreg"): "6pryn4us",
    MaskRun(radius=0.12,     length=1,  framework="jepa_noreg"): "khnj6and",
    MaskRun(radius=0.12,     length=2,  framework="jepa_noreg"): "mtbmkrzi",
    MaskRun(radius=0.12,     length=4,  framework="jepa_noreg"): "fw933z81",
    MaskRun(radius=0.12,     length=8,  framework="jepa_noreg"): "b5zaymrl",
    MaskRun(radius=0.12,     length=16, framework="jepa_noreg"): "ximmpkge",
    MaskRun(radius=0.12,     length=33, framework="jepa_noreg"): "m9jztpss",
    MaskRun(radius=math.inf, length=1,  framework="jepa_noreg"): "l29shjdg",
    MaskRun(radius=math.inf, length=4,  framework="jepa_noreg"): "08rlsatl",
    MaskRun(radius=math.inf, length=8,  framework="jepa_noreg"): "1mc641uc",
    MaskRun(radius=math.inf, length=16, framework="jepa_noreg"): "em0csd3m",
}

assert len(MASK_SWEEP_RUN_IDS) == 58
assert len(set(MASK_SWEEP_RUN_IDS.values())) == 58


