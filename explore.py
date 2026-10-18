"""Exploratory analyses (not pre-registered). See NOTES.md for why each was run.

    python explore.py --run main     # -> results/<run>/explore.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import monitors as M

ROOT = Path(__file__).resolve().parent
CONDS = ("ref", "w8", "w4", "steps2")
ALWAYS_FAIL = (0, 4)  # tasks with 0/50 reference successes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    eps = M.load_run(ROOT / "data" / args.run)
    mons = M.build_monitors(M.select(eps, "ref", M.TRAIN))
    calib = M.select(eps, "ref", M.CALIB)
    out = {"excl_always_fail_tasks": {}, "score_quantiles_failures": {}}
    for name, mon in mons.items():
        tau = M.conformal_tau([M.episode_score(mon.steps(e)) for e in calib if e.success])
        for c in CONDS:
            test = [e for e in M.select(eps, c, M.TEST) if e.task_id not in ALWAYS_FAIL]
            r = M.evaluate(mon, test, tau)
            out["excl_always_fail_tasks"].setdefault(name, {})[c] = {k: r[k] for k in ("n_success", "n_fail", "far", "tpr", "auroc")}
            fails = [M.episode_score(mon.steps(e)) for e in M.select(eps, c, M.TEST) if not e.success]
            out["score_quantiles_failures"].setdefault(name, {})[c] = {
                "tau": tau, "q10_50_90": [float(np.quantile(fails, q)) for q in (0.1, 0.5, 0.9)] if fails else None}
    path = ROOT / "results" / args.run / "explore.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    for name, per in out["excl_always_fail_tasks"].items():
        print(name, {c: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()} for c, d in per.items()})
    for name, per in out["score_quantiles_failures"].items():
        print(name, "tau", round(per["ref"]["tau"], 3), {c: [round(x, 3) for x in d["q10_50_90"]] for c, d in per.items() if d["q10_50_90"]})


if __name__ == "__main__":
    main()
