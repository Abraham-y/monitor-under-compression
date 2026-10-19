"""Exploratory analyses (not pre-registered). See NOTES.md for why each was run.

    python explore.py --run main     # -> results/<run>/explore.json

- excl_always_fail_tasks: drop tasks 0 and 4 (0/50 ref successes) from the test set.
- within_task_auroc: AUROC computed inside each task, then averaged over tasks with both
  outcomes, so a probe cannot score by recognising the task (review finding, NOTES.md).
- fixed_horizon: alarms and calibration use only the first H steps of every episode, since
  failures run to the 520-step timeout and successes stop early.
- matched_t0: step-0 representation shift on tasks where ref and steps2 saw bit-identical
  step-0 VLM states (tasks whose ref runs saw different first observations are excluded).
- score_quantiles_failures: where failed episodes' scores sit relative to tau.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import monitors as M

ROOT = Path(__file__).resolve().parent
CONDS = ("ref", "w8", "w4", "steps2", "w4g32")
ALWAYS_FAIL = (0, 4)  # tasks with 0/50 reference successes
HORIZONS = (200, 300)


def _truncate(e: M.Episode, h: int) -> M.Episode:
    return M.Episode(e.condition, e.task_id, e.init, e.seed, e.success, e.h_vlm[:h], e.h_exp[:h], e.chunk[:h])


def within_task_auroc(mon, test) -> tuple[float, int]:
    aucs = []
    for t in sorted({e.task_id for e in test}):
        sel = [e for e in test if e.task_id == t]
        lab = [not e.success for e in sel]
        if len(set(lab)) == 2:
            aucs.append(roc_auc_score(lab, [M.episode_score(mon.steps(e)) for e in sel]))
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def matched_tasks(eps) -> list[int]:
    ref = {(e.task_id, e.init): e for e in eps if e.condition == "ref"}
    ok = []
    for t in sorted({e.task_id for e in eps}):
        pairs = [(ref[(e.task_id, e.init)], e) for e in eps
                 if e.condition == "steps2" and e.task_id == t and (e.task_id, e.init) in ref]
        if pairs and all(np.array_equal(a.h_vlm[0], b.h_vlm[0]) for a, b in pairs):
            ok.append(t)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    eps = M.load_run(ROOT / "data" / args.run)
    conds = [c for c in CONDS if any(e.condition == c for e in eps)]
    mons = M.build_monitors(M.select(eps, "ref", M.TRAIN))
    calib = M.select(eps, "ref", M.CALIB)
    out = {"excl_always_fail_tasks": {}, "within_task_auroc": {}, "fixed_horizon": {},
           "score_quantiles_failures": {}}
    for name, mon in mons.items():
        tau = M.conformal_tau([M.episode_score(mon.steps(e)) for e in calib if e.success])
        for c in conds:
            test_all = M.select(eps, c, M.TEST)
            test = [e for e in test_all if e.task_id not in ALWAYS_FAIL]
            r = M.evaluate(mon, test, tau)
            out["excl_always_fail_tasks"].setdefault(name, {})[c] = {k: r[k] for k in ("n_success", "n_fail", "far", "tpr", "auroc")}
            auc, n_tasks = within_task_auroc(mon, test_all)
            out["within_task_auroc"].setdefault(name, {})[c] = {"auroc": auc, "n_tasks": n_tasks}
            fails = [M.episode_score(mon.steps(e)) for e in test_all if not e.success]
            out["score_quantiles_failures"].setdefault(name, {})[c] = {
                "tau": tau, "q10_50_90": [float(np.quantile(fails, q)) for q in (0.1, 0.5, 0.9)] if fails else None}
            for h in HORIZONS:
                cal_h = [_truncate(e, h) for e in calib if e.success]
                tau_h = M.conformal_tau([M.episode_score(mon.steps(e)) for e in cal_h])
                rh = M.evaluate(mon, [_truncate(e, h) for e in test_all], tau_h)
                out["fixed_horizon"].setdefault(name, {}).setdefault(str(h), {})[c] = {
                    k: rh[k] for k in ("n_success", "n_fail", "far", "tpr")}

    tasks = matched_tasks(eps)
    out["matched_t0"] = {"tasks": tasks}
    ref = {(e.task_id, e.init): e for e in eps if e.condition == "ref" and e.task_id in tasks}
    for key in ("h_vlm", "h_exp"):
        spread = []
        for t in tasks:
            H = np.stack([e.__dict__[key][0].astype(np.float32) for (tt, _), e in ref.items() if tt == t])
            d = np.linalg.norm(H[:, None] - H[None], axis=-1)
            spread.append(d[np.triu_indices(len(H), 1)].mean())
        spread = float(np.mean(spread))
        for c in conds[1:]:
            ds = [np.linalg.norm(ref[(e.task_id, e.init)].__dict__[key][0].astype(np.float32)
                                 - e.__dict__[key][0].astype(np.float32))
                  for e in eps if e.condition == c and (e.task_id, e.init) in ref]
            if ds:
                out["matched_t0"].setdefault(c, {})[key] = {"rel_dist": float(np.mean(ds)) / spread, "n": len(ds)}

    path = ROOT / "results" / args.run / "explore.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "score_quantiles_failures"}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
