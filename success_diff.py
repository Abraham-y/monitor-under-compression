"""Paired task-success changes against the original policy, for the camera-ready (exploratory).

    python success_diff.py --run main      # -> results/main/success_diff.txt

Compares 2 steps with the original (run main) and, when data/schedule exists, the original
executing 10 actions per plan (ref_a10) with the original as shipped, which replans every step.

The interval bootstraps tasks (10,000 resamples). Discordant counts are initial states that
succeed under one condition only; p is an exact McNemar test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import monitors as M

ROOT = Path(__file__).resolve().parent
N_BOOT = 10_000


def paired_diff(eps, inits, rng, other: str = "steps2") -> str:
    a = {(e.task_id, e.init): e.success for e in M.select(eps, "ref", set(inits))}
    b = {(e.task_id, e.init): e.success for e in M.select(eps, other, set(inits))}
    keys = sorted(set(a) & set(b))
    x = np.array([[a[k], b[k]] for k in keys], float)
    tasks = np.array([k[0] for k in keys])
    ut = np.unique(tasks)
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.where(tasks == t)[0] for t in rng.choice(ut, len(ut))])
        boots.append(x[idx, 1].mean() - x[idx, 0].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    gain = int(((x[:, 1] == 1) & (x[:, 0] == 0)).sum())
    loss = int(((x[:, 1] == 0) & (x[:, 0] == 1)).sum())
    p = binomtest(gain, gain + loss, 0.5).pvalue if gain + loss else 1.0
    return (f"n={len(keys)} ref={x[:, 0].mean():.3f} {other}={x[:, 1].mean():.3f} "
            f"diff={x[:, 1].mean() - x[:, 0].mean():+.3f} task-bootstrap 95% [{lo:+.3f}, {hi:+.3f}] "
            f"{other}-only {gain}, ref-only {loss}, McNemar p={p:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    eps = M.load_run(ROOT / "data" / args.run)
    rng = np.random.default_rng(0)
    lines = [f"{name}: {paired_diff(eps, inits, rng)}"
             for name, inits in [("test 40-49", range(40, 50)), ("calib+test 30-49", range(30, 50))]]
    if (ROOT / "data" / "schedule").exists():
        eps_a10 = [e for e in M.load_run(ROOT / "data" / "schedule") if e.condition == "ref_a10"]
        ref_test = M.select(eps, "ref", set(range(40, 50)))
        lines.append(f"ref_a10 (10 actions/plan) vs ref, test 40-49: "
                     f"{paired_diff(ref_test + eps_a10, range(40, 50), rng, 'ref_a10')}")
    out = ROOT / "results" / args.run / "success_diff.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
