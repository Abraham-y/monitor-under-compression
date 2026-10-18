"""Figures for the paper from results/<run>/results.json and the raw episodes.

    python figures.py --run main      # -> paper/fig_far.pdf, paper/fig_scores.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import monitors as M

ROOT = Path(__file__).resolve().parent
CONDS = ("ref", "w8", "w4", "steps2")
COND_LABEL = {"ref": "reference", "w8": "int8 W", "w4": "int4 W", "steps2": "2 flow steps"}
COLORS = {"ref": "#4a4a4a", "w8": "#5b8fd6", "w4": "#d0573a", "steps2": "#3f9e6e"}
MON_LABEL = {
    "probe_vlm": "probe (VLM state)",
    "probe_exp": "probe (expert state)",
    "maha_vlm": "Mahalanobis (VLM)",
    "act_cons": "action consistency",
}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def fig_far(res: dict, path: Path):
    mons = list(res["monitors"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), gridspec_kw={"width_ratios": [3.2, 1]})
    ax = axes[0]
    w = 0.19
    x = np.arange(len(mons))
    for j, c in enumerate(CONDS):
        far = [res["monitors"][m].get(c, {}).get("far", np.nan) for m in mons]
        ci = [res["monitors"][m].get(c, {}).get("far_ci", (np.nan, np.nan)) for m in mons]
        lo = [f - c_[0] for f, c_ in zip(far, ci)]
        hi = [c_[1] - f for f, c_ in zip(far, ci)]
        ax.bar(x + (j - 1.5) * w, far, w, color=COLORS[c], label=COL_LABEL(c),
               yerr=[lo, hi], error_kw={"lw": 0.7, "capsize": 1.5})
    ax.axhline(res["alpha"], color="k", ls="--", lw=0.8)
    ax.text(len(mons) - 0.5, res["alpha"], r" nominal $\alpha$", va="bottom", ha="right", fontsize=7)
    ax.set_xticks(x, [MON_LABEL[m] for m in mons])
    ax.set_ylabel("false-alarm rate\n(successful test episodes)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    s = [res["success"][c]["test"]["success"] for c in CONDS]
    ci = [res["success"][c]["test"]["ci"] for c in CONDS]
    ax.bar(range(len(CONDS)), s, color=[COLORS[c] for c in CONDS],
           yerr=[[v - c_[0] for v, c_ in zip(s, ci)], [c_[1] - v for v, c_ in zip(s, ci)]],
           error_kw={"lw": 0.7, "capsize": 1.5})
    ax.set_xticks(range(len(CONDS)), [COL_LABEL(c) for c in CONDS], rotation=30, ha="right")
    ax.set_ylabel("task success (test)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def COL_LABEL(c: str) -> str:
    return COND_LABEL[c]


def fig_scores(run: str, res: dict, path: Path, monitor: str):
    """Episode scores of successful test episodes per condition vs the deployed tau."""
    eps = M.load_run(ROOT / "data" / run)
    train = M.select(eps, "ref", M.TRAIN)
    mon = M.build_monitors(train)[monitor]
    tau = res["monitors"][monitor]["tau_ref"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for j, c in enumerate(CONDS):
        sc = [M.episode_score(mon.steps(e)) for e in M.select(eps, c, M.TEST) if e.success]
        if not sc:
            continue
        jit = np.random.default_rng(j).uniform(-0.15, 0.15, len(sc))
        ax.scatter(np.full(len(sc), j) + jit, sc, s=6, color=COLORS[c], alpha=0.7, lw=0)
    ax.axhline(tau, color="k", ls="--", lw=0.8)
    ax.text(len(CONDS) - 0.6, tau, r" $\tau$ (ref calib.)", va="bottom", ha="right", fontsize=7)
    ax.set_xticks(range(len(CONDS)), [COL_LABEL(c) for c in CONDS])
    ax.set_ylabel(f"episode score\n{MON_LABEL[monitor]}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--score-monitor", default="maha_vlm")
    args = ap.parse_args()
    res = json.loads((ROOT / "results" / args.run / "results.json").read_text())
    out = ROOT / "paper"
    fig_far(res, out / "fig_far.pdf")
    fig_scores(args.run, res, out / "fig_scores.pdf", args.score_monitor)
    print("wrote", out / "fig_far.pdf", out / "fig_scores.pdf")


if __name__ == "__main__":
    main()
