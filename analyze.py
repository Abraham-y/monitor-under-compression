"""Fit monitors on reference train episodes; score every condition's test episodes.

    python analyze.py --run main            # -> results/main/{results.json, table.md}

Implements PREREG.md: the deployed threshold comes from successful ref calibration
episodes; the recalibrated one from the condition's own calibration episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import monitors as M

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("ref", "w8", "w4", "steps2")
N_BOOT = 10_000


def success_table(eps) -> dict:
    out = {}
    for c in CONDITIONS:
        for split, inits in (("calib", M.CALIB), ("test", M.TEST), ("all", range(50))):
            sel = M.select(eps, c, inits)
            if not sel:
                continue
            k = sum(e.success for e in sel)
            out.setdefault(c, {})[split] = {"n": len(sel), "success": k / len(sel), "ci": M.wilson(k, len(sel))}
    return out


def boot_far_diff(mon, a_eps, b_eps, tau, rng) -> tuple[float, tuple[float, float]]:
    """FAR(a) - FAR(b), successful episodes resampled within each condition."""
    fa = np.array([M.first_alarm(mon.steps(e), tau) is not None for e in a_eps if e.success], float)
    fb = np.array([M.first_alarm(mon.steps(e), tau) is not None for e in b_eps if e.success], float)
    if len(fa) == 0 or len(fb) == 0:
        return float("nan"), (float("nan"), float("nan"))
    d = [rng.choice(fa, len(fa)).mean() - rng.choice(fb, len(fb)).mean() for _ in range(N_BOOT)]
    return float(fa.mean() - fb.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    eps = M.load_run(ROOT / "data" / args.run)
    out_dir = ROOT / "results" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    res = {"success": success_table(eps), "monitors": {}, "alpha": M.ALPHA}
    train = M.select(eps, "ref", M.TRAIN)
    res["n_train"] = {"episodes": len(train), "failures": sum(not e.success for e in train)}
    mons = M.build_monitors(train, args.seed)
    res["probe_C"] = {k: {"C": m.C, "cv_auroc": m.cv} for k, m in mons.items() if hasattr(m, "C")}

    ref_calib = M.select(eps, "ref", M.CALIB)
    ref_test = M.select(eps, "ref", M.TEST)
    for name, mon in mons.items():
        tau_ref = M.conformal_tau([M.episode_score(mon.steps(e)) for e in ref_calib if e.success])
        per = {"tau_ref": tau_ref}
        for c in CONDITIONS:
            test = M.select(eps, c, M.TEST)
            if not test:
                continue
            r = M.evaluate(mon, test, tau_ref)
            k = round(r["far"] * r["n_success"])
            r["far_binom_p_gt_alpha"] = (
                float(binomtest(k, r["n_success"], M.ALPHA, alternative="greater").pvalue) if r["n_success"] else None
            )
            own = M.select(eps, c, M.CALIB)
            tau_own = M.conformal_tau([M.episode_score(mon.steps(e)) for e in own if e.success])
            r["recal"] = {"tau": tau_own, **{k2: v for k2, v in M.evaluate(mon, test, tau_own).items()
                                             if k2 in ("far", "far_ci", "tpr", "tpr_ci")}}
            if c != "ref":
                r["far_minus_ref"], r["far_minus_ref_ci"] = boot_far_diff(mon, test, ref_test, tau_ref, rng)
            per[c] = r
        res["monitors"][name] = per
        print(name, json.dumps({c: {k: per[c][k] for k in ("far", "tpr", "auroc")} for c in CONDITIONS if c in per}))

    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float))
    write_table(res, out_dir / "table.md")


def write_table(res: dict, path: Path):
    lines = ["| monitor | condition | success (test) | FAR [95% CI] | p(FAR>α) | TPR | AUROC | FAR recal. |",
             "|---|---|---|---|---|---|---|---|"]
    for name, per in res["monitors"].items():
        for c in CONDITIONS:
            if c not in per:
                continue
            r = per[c]
            s = res["success"][c]["test"]["success"]
            lo, hi = r["far_ci"]
            p = r.get("far_binom_p_gt_alpha")
            lines.append(
                f"| {name} | {c} | {s:.2f} | {r['far']:.2f} [{lo:.2f}, {hi:.2f}] | {p:.3f} | "
                f"{r['tpr']:.2f} | {r['auroc']:.3f} | {r['recal']['far']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
