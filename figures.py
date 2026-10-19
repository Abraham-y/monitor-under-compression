"""Figure 1 for the paper from results/<run>/{results,robustness}.json and the raw episodes.

    python figures.py --run main      # -> paper/fig_main.pdf (+ .png preview)
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
CONDS = ("ref", "w8", "steps2", "w4g32", "w4")
COND_LABEL = {"ref": "reference", "w8": "int8", "steps2": "2 flow steps", "w4g32": "int4-g32",
              "w4": "int4-coarse"}
SHORT = {"ref": "ref", "w8": "int8", "steps2": "2 steps", "w4g32": "int4-g32", "w4": "int4-c"}
COLORS = {"ref": "#4a4a4a", "w8": "#5b8fd6", "steps2": "#3f9e6e", "w4g32": "#e09a3c", "w4": "#d0573a"}
MONS = ("act_cons", "probe_exp", "probe_vlm", "maha_vlm")
MON_LABEL = {"act_cons": "action\nconsist.", "probe_exp": "probe\n(expert)", "probe_vlm": "probe\n(VLM)",
             "maha_vlm": "Mahal.\n(VLM)"}

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    rob = json.loads((ROOT / "results" / args.run / "robustness.json").read_text())
    res = json.loads((ROOT / "results" / args.run / "results.json").read_text())
    conds = [c for c in CONDS if c in rob["conditions"] and c in res["success"]]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0), gridspec_kw={"width_ratios": [0.9, 2.6, 1.5]})

    ax = axes[0]
    s = np.array([res["success"][c]["test"]["success"] for c in conds])
    ci = np.array([res["success"][c]["test"]["ci"] for c in conds])
    ax.bar(range(len(conds)), s, color=[COLORS[c] for c in conds],
           yerr=[s - ci[:, 0], ci[:, 1] - s], error_kw={"lw": 0.6, "capsize": 1.2})
    ax.set_xticks(range(len(conds)), [SHORT[c] for c in conds], rotation=50, ha="right", rotation_mode="anchor")
    ax.set_ylabel("task success")
    ax.set_ylim(0, 1)
    ax.set_title("(a) policy", fontsize=7.5)

    ax = axes[1]
    w = 0.8 / len(conds)
    x = np.arange(len(MONS))
    for j, c in enumerate(conds):
        tpr, lo, hi = [], [], []
        for m in MONS:
            r = rob["table"][m][c]
            a, b = M.wilson(round(r["tpr"] * r["n_fail"]), r["n_fail"])
            tpr.append(r["tpr"]); lo.append(r["tpr"] - a); hi.append(b - r["tpr"])
        xs = x + (j - (len(conds) - 1) / 2) * w
        ax.bar(xs, tpr, w, color=COLORS[c], label=COND_LABEL[c], hatch="////" if c == "w4" else None,
               edgecolor="white" if c == "w4" else None, lw=0, yerr=[lo, hi], error_kw={"lw": 0.6, "capsize": 1.0})
        for xi, v in zip(xs, tpr):
            if v == 0:
                ax.text(xi, 0.03, "0", ha="center", va="bottom", fontsize=6, color=COLORS[c], fontweight="bold")
    ax.set_xticks(x, [MON_LABEL[m] for m in MONS])
    ax.set_ylabel("failures caught\n(inherited threshold)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) detection", fontsize=7.5)
    handles, labels = ax.get_legend_handles_labels()

    ax = axes[2]
    eps = M.load_run(ROOT / "data" / args.run)
    mon = M.ActionConsistency()
    calib = M.select(eps, "ref", M.CALIB)
    tau = M.conformal_tau([M.episode_score(mon.steps(e)) for e in calib if e.success])
    for j, c in enumerate(conds):
        sc = [M.episode_score(mon.steps(e)) for e in M.select(eps, c, M.TEST) if not e.success]
        jit = np.random.default_rng(j).uniform(-0.18, 0.18, len(sc))
        ax.scatter(np.full(len(sc), j) + jit, sc, s=2.5, color=COLORS[c], alpha=0.6, lw=0)
    ax.axhline(tau, color="k", ls="--", lw=0.6)
    ax.set_xticks(range(len(conds)), [SHORT[c] for c in conds], rotation=50, ha="right", rotation_mode="anchor")
    ax.set_ylabel("episode score,\nfailed episodes")
    ax.set_title("(c) action consistency", fontsize=7.5)

    fig.tight_layout(w_pad=0.6)
    fig.legend(handles, labels, ncol=len(conds), frameon=False, fontsize=6.8, loc="lower center",
               bbox_to_anchor=(0.5, 0.97), handlelength=1.1, columnspacing=1.0)
    out = ROOT / "paper" / "fig_main.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print("wrote", out)


if __name__ == "__main__":
    main()
