"""Figures for the paper from results/<run>/results.json and the raw episodes.

    python figures.py --run main      # -> paper/fig_main.pdf, paper/fig_scores.pdf
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
CONDS = ("ref", "w8", "w4", "steps2", "w4nv")
COND_LABEL = {"ref": "reference", "w8": "int8 W", "w4": "int4 W", "steps2": "2 flow steps"}
COLORS = {"ref": "#4a4a4a", "w8": "#5b8fd6", "w4": "#d0573a", "steps2": "#3f9e6e"}
MON_LABEL = {
    "probe_vlm": "probe\n(VLM)",
    "probe_exp": "probe\n(expert)",
    "maha_vlm": "Mahal.\n(VLM)",
    "act_cons": "action\nconsist.",
}

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6})


def _bars(ax, res, key, ci_key, ylabel):
    mons = list(res["monitors"])
    w = 0.2
    x = np.arange(len(mons))
    for j, c in enumerate(CONDS):
        v = np.array([res["monitors"][m][c][key] for m in mons])
        ci = np.array([res["monitors"][m][c][ci_key] for m in mons])
        hatch = "////" if c == "w4" else None
        for xi, vi in zip(x + (j - 1.5) * w, v):
            if vi == 0 and key == "tpr":
                ax.text(xi, 0.06, "0", ha="center", va="bottom", fontsize=6, color=COLORS[c], fontweight="bold")
        ax.bar(x + (j - 1.5) * w, v, w, color=COLORS[c], label=COND_LABEL[c], hatch=hatch,
               edgecolor="white" if hatch else None, lw=0,
               yerr=[v - ci[:, 0], ci[:, 1] - v], error_kw={"lw": 0.6, "capsize": 1.2})
    ax.set_xticks(x, [MON_LABEL[m] for m in mons])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)


def fig_main(res: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95), gridspec_kw={"width_ratios": [0.8, 2.2, 2.2]})
    ax = axes[0]
    s = np.array([res["success"][c]["test"]["success"] for c in CONDS])
    ci = np.array([res["success"][c]["test"]["ci"] for c in CONDS])
    ax.bar(range(4), s, color=[COLORS[c] for c in CONDS], yerr=[s - ci[:, 0], ci[:, 1] - s],
           error_kw={"lw": 0.6, "capsize": 1.2})
    ax.set_xticks(range(4), [COND_LABEL[c] for c in CONDS], rotation=35, ha="right")
    ax.set_ylabel("task success")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) policy", fontsize=7.5)

    _bars(axes[1], res, "tpr", "tpr_ci", "detection rate\n(failed episodes)")
    axes[1].set_title("(b) failures caught, reference threshold", fontsize=7.5)
    _bars(axes[2], res, "far", "far_ci", "false-alarm rate\n(successful episodes)")
    axes[2].axhline(res["alpha"], color="k", ls="--", lw=0.6)
    axes[2].set_title(r"(c) false alarms (dashed: $\alpha$)", fontsize=7.5)
    mons = list(res["monitors"])
    for i, m in enumerate(mons):  # int4 keeps 1 successful test episode: flag its FAR bar
        n = res["monitors"][m]["w4"]["n_success"]
        axes[2].text(i + 0.5 * 0.2, 1.02, f"n={n}", ha="center", va="bottom", fontsize=5.5, color=COLORS["w4"])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.6, 1.08),
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=220)


def fig_scores(run: str, path: Path):
    """Episode scores of failed test episodes per condition vs the deployed tau."""
    eps = M.load_run(ROOT / "data" / run)
    mons = M.build_monitors(M.select(eps, "ref", M.TRAIN))
    calib = M.select(eps, "ref", M.CALIB)
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    for ax, name, title in ((axes[0], "probe_vlm", "probe (VLM state)"),
                            (axes[1], "act_cons", "action consistency")):
        mon = mons[name]
        tau = M.conformal_tau([M.episode_score(mon.steps(e)) for e in calib if e.success])
        for j, c in enumerate(CONDS):
            sc = [M.episode_score(mon.steps(e)) for e in M.select(eps, c, M.TEST) if not e.success]
            jit = np.random.default_rng(j).uniform(-0.18, 0.18, len(sc))
            ax.scatter(np.full(len(sc), j) + jit, sc, s=3, color=COLORS[c], alpha=0.6, lw=0)
        ax.axhline(tau, color="k", ls="--", lw=0.6)
        ax.set_xticks(range(4), [COND_LABEL[c] for c in CONDS], rotation=35, ha="right")
        ax.set_title(title, fontsize=7.5)
    axes[0].set_ylabel("episode score,\nfailed episodes")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=220)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    res = json.loads((ROOT / "results" / args.run / "results.json").read_text())
    out = ROOT / "paper"
    fig_main(res, out / "fig_main.pdf")
    fig_scores(args.run, out / "fig_scores.pdf")
    print("wrote", out / "fig_main.pdf", out / "fig_scores.pdf")


if __name__ == "__main__":
    main()
