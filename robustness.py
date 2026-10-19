"""Robustness analyses requested by the adversarial review (exploratory, not pre-registered).

    python robustness.py --run main      # -> results/<run>/robustness.{json,md}

1. Calibration vs separability, per monitor x condition:
     deployed TPR/FAR (ref threshold), recalibrated TPR/FAR (own calib successes, same
     alpha), oracle TPR at FAR <= alpha (threshold chosen on the test set itself),
     pooled AUROC, within-task AUROC.
2. Change in detection vs ref:
     paired McNemar on (task, init) pairs that fail in both conditions, and a bootstrap
     that resamples tasks (test) and ref calibration successes (tau) together.
3. Fixed horizon: alarms counted only before step 200.
4. Same-input dose response: at step 0 every condition sees the same observation on the
     tasks whose ref and steps2 step-0 VLM states are bit-identical; the score shift there
     isolates what compression does to each monitor's input.
5. act_cons rescaling: per (task, init), mean chunk disagreement over steps 1-30 in a
     condition divided by ref.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

import monitors as M

ROOT = Path(__file__).resolve().parent
CONDS = ("ref", "w8", "steps2", "w4g32", "w4")
N_BOOT = 4000
HORIZON = 200


def oracle_tpr(succ_scores, fail_scores, alpha):
    """Best TPR over thresholds whose FAR on these same successes is <= alpha."""
    if not succ_scores or not fail_scores:
        return float("nan")
    s = np.sort(succ_scores)
    k = int(np.floor(alpha * len(s)))  # allow k false alarms
    tau = s[len(s) - k - 1]
    return float(np.mean(np.array(fail_scores) > tau))


def detected(steps, tau, horizon=None):
    S = M.running_mean(steps if horizon is None else steps[:horizon])
    return bool((S[M.BURN_IN:] > tau).any()) if len(S) > M.BURN_IN else bool(S[-1] > tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    eps = M.load_run(ROOT / "data" / args.run)
    conds = [c for c in CONDS if any(e.condition == c for e in eps)]
    mons = M.build_monitors(M.select(eps, "ref", M.TRAIN))
    keep = [e for e in eps if e.init in M.CALIB or e.init in M.TEST]
    steps = {name: {id(e): mon.steps(e) for e in keep} for name, mon in mons.items()}
    score = {name: {k: M.episode_score(v) for k, v in d.items()} for name, d in steps.items()}

    out = {"alpha": M.ALPHA, "conditions": conds, "table": {}, "delta_tpr": {}, "dose_t0": {}, "act_cons_ratio": {}}
    ref_cal_succ = [e for e in M.select(eps, "ref", M.CALIB) if e.success]
    test = {c: M.select(eps, c, M.TEST) for c in conds}
    by_key = {c: {(e.task_id, e.init): e for e in test[c]} for c in conds}

    for name in mons:
        sc = score[name]
        tau_ref = M.conformal_tau([sc[id(e)] for e in ref_cal_succ])
        for c in conds:
            T = test[c]
            succ = [e for e in T if e.success]
            fail = [e for e in T if not e.success]
            own_cal = [e for e in M.select(eps, c, M.CALIB) if e.success]
            tau_own = M.conformal_tau([sc[id(e)] for e in own_cal]) if own_cal else float("inf")
            det = [detected(steps[name][id(e)], tau_ref) for e in fail]
            alarm_t = [M.first_alarm(steps[name][id(e)], tau_ref) for e in fail]
            row = {
                "n_success": len(succ), "n_fail": len(fail),
                "tpr": float(np.mean(det)) if fail else float("nan"),
                "far": float(np.mean([detected(steps[name][id(e)], tau_ref) for e in succ])) if succ else float("nan"),
                "tpr_h200": float(np.mean([detected(steps[name][id(e)], tau_ref, HORIZON) for e in fail])) if fail else float("nan"),
                "tpr_recal": float(np.mean([sc[id(e)] > tau_own for e in fail])) if fail and own_cal else float("nan"),
                "far_recal": float(np.mean([sc[id(e)] > tau_own for e in succ])) if succ and own_cal else float("nan"),
                "n_own_cal_success": len(own_cal),
                "median_alarm_step": float(np.median([t for t in alarm_t if t is not None])) if any(t is not None for t in alarm_t) else float("nan"),
            }
            if len(succ) >= 15 and fail:
                row["auroc"] = float(roc_auc_score([0] * len(succ) + [1] * len(fail), [sc[id(e)] for e in succ + fail]))
                row["oracle_tpr"] = oracle_tpr([sc[id(e)] for e in succ], [sc[id(e)] for e in fail], M.ALPHA)
            else:
                row["auroc"] = row["oracle_tpr"] = float("nan")  # too few successes to rank against
            wt = []
            for t in sorted({e.task_id for e in T}):
                sel = [e for e in T if e.task_id == t]
                lab = [not e.success for e in sel]
                if len(set(lab)) == 2:
                    wt.append(roc_auc_score(lab, [sc[id(e)] for e in sel]))
            row["within_task_auroc"] = float(np.mean(wt)) if wt and len(succ) >= 15 else float("nan")
            row["n_tasks_within"] = len(wt)
            out["table"].setdefault(name, {})[c] = row

            if c == "ref":
                continue
            # paired McNemar on pairs failing in both conditions
            pairs = [(by_key["ref"][k], by_key[c][k]) for k in by_key[c] if k in by_key["ref"]
                     and not by_key["ref"][k].success and not by_key[c][k].success]
            a = [detected(steps[name][id(r)], tau_ref) for r, _ in pairs]
            b = [detected(steps[name][id(x)], tau_ref) for _, x in pairs]
            lost = sum(x and not y for x, y in zip(a, b))
            gained = sum(y and not x for x, y in zip(a, b))
            p = float(binomtest(lost, lost + gained, 0.5).pvalue) if lost + gained else 1.0
            # bootstrap: tasks (test) and ref calib successes (tau) resampled together
            tasks = sorted({e.task_id for e in T})
            ref_fail_by_task = {t: [e for e in test["ref"] if e.task_id == t and not e.success] for t in tasks}
            c_fail_by_task = {t: [e for e in T if e.task_id == t and not e.success] for t in tasks}
            cal_scores = np.array([sc[id(e)] for e in ref_cal_succ])
            S_ref = {t: [steps[name][id(e)] for e in ref_fail_by_task[t]] for t in tasks}
            S_c = {t: [steps[name][id(e)] for e in c_fail_by_task[t]] for t in tasks}
            # precompute each failure's max running-mean score so detection is a comparison
            mx_ref = {t: [M.episode_score(s) for s in S_ref[t]] for t in tasks}
            mx_c = {t: [M.episode_score(s) for s in S_c[t]] for t in tasks}
            diffs = []
            for _ in range(N_BOOT):
                tau_b = M.conformal_tau(list(rng.choice(cal_scores, len(cal_scores))))
                tb = rng.choice(tasks, len(tasks))
                r = [x for t in tb for x in mx_ref[t]]
                q = [x for t in tb for x in mx_c[t]]
                if r and q:
                    diffs.append(np.mean(np.array(q) > tau_b) - np.mean(np.array(r) > tau_b))
            out["delta_tpr"].setdefault(name, {})[c] = {
                "delta": row["tpr"] - out["table"][name]["ref"]["tpr"],
                "ci95_task_and_tau_bootstrap": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "mcnemar_pairs_both_fail": len(pairs), "lost": lost, "gained": gained, "mcnemar_p": p,
            }

    # same-input dose response at step 0, on tasks where ref and steps2 match bit for bit
    ref_k = {(e.task_id, e.init): e for e in eps if e.condition == "ref"}
    s2 = {(e.task_id, e.init): e for e in eps if e.condition == "steps2"}
    matched = sorted({t for (t, _), e in s2.items()} - {t for (t, i), e in s2.items()
                                                         if (t, i) in ref_k and not np.array_equal(ref_k[(t, i)].h_vlm[0], e.h_vlm[0])})
    out["dose_t0"]["matched_tasks"] = matched
    for name, mon in mons.items():
        if name == "act_cons":
            continue
        base = {k: float(mon.steps(_first(e))[0]) for k, e in ref_k.items() if k[1] >= 30 and k[0] in matched}
        sd = float(np.std(list(base.values())))
        for c in conds[1:]:
            d = [float(mon.steps(_first(e))[0]) - base[(e.task_id, e.init)] for e in eps
                 if e.condition == c and (e.task_id, e.init) in base]
            if d:
                sign = -1 if name.startswith("probe") else -1  # lower score = "looks safe" for all three
                out["dose_t0"].setdefault(name, {})[c] = {
                    "mean_shift_sd": float(np.mean(d)) / sd, "frac_toward_safe": float(np.mean(np.sign(d) == sign)),
                    "n": len(d)}

    # act_cons rescaling over steps 1-30
    for c in conds[1:]:
        r = []
        for e in eps:
            if e.condition == c and (e.task_id, e.init) in ref_k:
                a = M.chunk_disagreement(ref_k[(e.task_id, e.init)].chunk[:31])[1:].mean()
                b = M.chunk_disagreement(e.chunk[:31])[1:].mean()
                r.append(b / a)
        if r:
            out["act_cons_ratio"][c] = {"median": float(np.median(r)), "iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))], "n": len(r)}

    d = ROOT / "results" / args.run
    (d / "robustness.json").write_text(json.dumps(out, indent=2, default=float))
    write_md(out, d / "robustness.md")


def _first(e: M.Episode) -> M.Episode:
    return M.Episode(e.condition, e.task_id, e.init, e.seed, e.success, e.h_vlm[:1], e.h_exp[:1], e.chunk[:1])


def f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_md(out, path):
    L = ["## Calibration vs separability (test split; deployed = ref-calibrated tau)", "",
         "| monitor | cond | succ/fail | TPR deployed | FAR deployed | TPR recal | FAR recal | oracle TPR@FAR≤α | AUROC | within-task AUROC | TPR (t<200) | median alarm step |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, per in out["table"].items():
        for c, r in per.items():
            L.append(f"| {name} | {c} | {r['n_success']}/{r['n_fail']} | {f(r['tpr'])} | {f(r['far'])} | {f(r['tpr_recal'])} | "
                     f"{f(r['far_recal'])} | {f(r['oracle_tpr'])} | {f(r['auroc'])} | {f(r['within_task_auroc'])} | "
                     f"{f(r['tpr_h200'])} | {f(r['median_alarm_step'], 0)} |")
    L += ["", "## Change in detection vs ref (deployed threshold)", "",
          "| monitor | cond | ΔTPR | 95% CI (tasks + tau bootstrap) | McNemar lost/gained (pairs) | p |", "|---|---|---|---|---|---|"]
    for name, per in out["delta_tpr"].items():
        for c, r in per.items():
            lo, hi = r["ci95_task_and_tau_bootstrap"]
            L.append(f"| {name} | {c} | {r['delta']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['lost']}/{r['gained']} ({r['mcnemar_pairs_both_fail']}) | {r['mcnemar_p']:.3g} |")
    L += ["", f"## Same-input score shift at step 0 (matched tasks {out['dose_t0']['matched_tasks']}), in ref SDs", "",
          "| monitor | " + " | ".join(out["conditions"][1:]) + " |", "|---|" + "---|" * (len(out["conditions"]) - 1)]
    for name, per in out["dose_t0"].items():
        if name == "matched_tasks":
            continue
        L.append(f"| {name} | " + " | ".join(
            f"{per[c]['mean_shift_sd']:+.2f} ({per[c]['frac_toward_safe']:.0%} lower)" if c in per else "—"
            for c in out["conditions"][1:]) + " |")
    L += ["", "## act_cons: chunk disagreement over steps 1-30, condition / ref (same task, init)", ""]
    for c, r in out["act_cons_ratio"].items():
        L.append(f"- {c}: median {r['median']:.2f} (IQR {r['iqr'][0]:.2f}-{r['iqr'][1]:.2f}, n={r['n']})")
    path.write_text("\n".join(L) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
