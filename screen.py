"""Which LIBERO-90 tasks can pi0.5 not do? Task selection and success screen (exploratory).

    python screen.py select    # tasks.json, training_tasks.txt -> selection.json
    python screen.py report    # data/libero90screen/ -> results/libero90screen/screen.txt
    python screen.py followup  # what the pooled AUROC is made of -> results/libero90screen/followup.txt
    python screen.py compare   # 10 vs 2 flow steps on the same episodes -> results/libero90screen/compare.txt

The pi0.5 LIBERO checkpoint trained on 40 tasks (Spatial, Object, Goal, Long), so LIBERO-90 is
outside its training set. The screen takes every third task, drops any whose instruction
matches a training instruction up to the wording differences in SAME, and runs init states
40-49 at 10 flow steps, 10 actions per plan. The decision rule was written down before any
rollout (NOTES.md): a full 10-vs-2-step test is worthwhile only if success is 20-80% and the
action-consistency score separates failures from successes (pooled AUROC >= 0.60).
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import monitors as M
import shrink

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "libero90screen"
RUN, COND, SUITE, OFFSET = "libero90screen", "pi05", "libero_90", 10
SAME = (("put it in", "place it in"), ("cream cheese box", "cream cheese"), ("wine rack", "rack"),
        ("black bowl", "bowl"))
MIN_SUCCESS, MAX_SUCCESS, MIN_AUROC = 0.20, 0.80, 0.60


def norm(s: str) -> str:
    for a, b in SAME:
        s = s.replace(a, b)
    return s


def select() -> None:
    tasks = json.loads((OUT / "tasks.json").read_text())
    train = {norm(x.strip()) for x in (OUT / "training_tasks.txt").read_text().splitlines() if x.strip()}
    keep, drop = [], []
    for t in tasks[::3]:
        (drop if norm(t["language"]) in train else keep).append(t)
    out = {"task_ids": [t["task_id"] for t in keep], "language": {t["task_id"]: t["language"] for t in keep},
           "dropped": {t["task_id"]: t["language"] for t in drop}}
    (OUT / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(keep)} tasks kept, {len(drop)} dropped: {sorted(out['dropped'])}")
    print(",".join(map(str, out["task_ids"])))


def report() -> None:
    sel = json.loads((OUT / "selection.json").read_text())
    eps = shrink.load(RUN, COND, OFFSET, SUITE)
    tasks = sorted({k[0] for k in eps})
    missing = sorted(set(sel["task_ids"]) - set(tasks))
    rate = {t: np.mean([e["success"] for k, e in eps.items() if k[0] == t]) for t in tasks}
    overall = float(np.mean([e["success"] for e in eps.values()]))
    rng = np.random.default_rng(0)
    boot = [np.mean([rate[t] for t in rng.choice(tasks, len(tasks))]) for _ in range(5000)]
    lines = [f"pi0.5 on LIBERO-90: {len(eps)} episodes, {len(tasks)} tasks, init states 40-49, seed 0",
             f"missing tasks: {missing}" if missing else "all selected tasks present",
             f"success {overall:.2f} (95% over tasks {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f}); "
             f"{sum(e['success'] for e in eps.values())} successes, {sum(not e['success'] for e in eps.values())} failures"]
    bins = [("0/10", 0, 0), ("1-3/10", .1, .3), ("4-6/10", .4, .6), ("7-9/10", .7, .9), ("10/10", 1, 1)]
    lines.append("tasks by success: " + ", ".join(f"{n}: {sum(a - 1e-9 <= rate[t] <= b + 1e-9 for t in tasks)}" for n, a, b in bins))

    y = [not e["success"] for e in eps.values()]
    s = [e["score"] for e in eps.values()]
    auroc = float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")
    within = []
    for t in tasks:
        sub = [e for k, e in eps.items() if k[0] == t]
        lab = [not e["success"] for e in sub]
        if len(set(lab)) == 2:
            within.append(roc_auc_score(lab, [e["score"] for e in sub]))
    lines.append(f"action-consistency AUROC (failure vs success): pooled {auroc:.2f}; within task, mean "
                 f"{np.mean(within) if within else float('nan'):.2f} over the {len(within)} tasks with both outcomes")
    usable = MIN_SUCCESS <= overall <= MAX_SUCCESS and auroc >= MIN_AUROC
    lines.append(f"decision rule (success {MIN_SUCCESS:.0%}-{MAX_SUCCESS:.0%}, pooled AUROC >= {MIN_AUROC}): "
                 f"{'full test worthwhile' if usable else 'not met'}")
    lines.append("")
    for t in sorted(tasks, key=lambda t: (rate[t], t)):
        lines.append(f"{rate[t]:.1f}  task {t:2d}  {sel['language'][str(t)]}")
    (OUT / "screen.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def per_call(run: str, suite: str, cond: str = COND) -> dict:
    eps = {}
    for f in sorted((ROOT / "data" / run / cond / suite).glob("*_init40-49_s0.npz")):
        z = np.load(f, allow_pickle=False)
        task = int(ast.literal_eval(str(z["meta"]))["task_id"])
        for b, init in enumerate(z["init_ids"]):
            n = -(-int(z["length"][b]) // OFFSET)
            eps[(task, int(init))] = (bool(z["success"][b]), int(z["length"][b]), M.chunk_disagreement(z["chunk"][b, :n], OFFSET))
    return eps


def episode_score(s, calls=None) -> float:
    S = M.running_mean(s[:calls])
    return float(S[1:].max()) if len(S) > 1 else float(S[-1])


def followup() -> None:
    """Added after seeing the screen: the pooled AUROC was far above the within-task one."""
    eps = per_call(RUN, SUITE)
    keys = sorted(eps)
    y = np.array([not eps[k][0] for k in keys])
    task = np.array([k[0] for k in keys])
    length = np.array([eps[k][1] for k in keys])
    full = np.array([episode_score(eps[k][2]) for k in keys])
    lines = [f"pooled AUROC {roc_auc_score(y, full):.2f}",
             f"episode length: successes median {int(np.median(length[~y]))} (range {length[~y].min()}-{length[~y].max()}), "
             f"failures all {length[y].min()}-{length[y].max()}",
             "pooled AUROC on the first T timesteps of every episode: "
             + ", ".join(f"T={c * OFFSET}: {roc_auc_score(y, [episode_score(eps[k][2], c) for k in keys]):.2f}" for c in (3, 5, 10, 20, 30))]
    mean = {t: full[task == t].mean() for t in set(task)}
    lines.append(f"pooled AUROC with each episode's score replaced by its task's mean: {roc_auc_score(y, [mean[t] for t in task]):.2f}")

    mixed = sorted(t for t in set(task) if 0 < y[task == t].sum() < 10)
    within = lambda lab: np.mean([roc_auc_score(lab[task == t], full[task == t]) for t in mixed])
    rng = np.random.default_rng(0)
    null = []
    for _ in range(5000):
        p = y.copy()
        for t in mixed:
            i = np.where(task == t)[0]
            p[i] = rng.permutation(p[i])
        null.append(within(p))
    sub = np.isin(task, mixed)
    lines.append(f"within-task AUROC: mean {within(y):.2f} over {len(mixed)} tasks ({(~y[sub]).sum()} successes, {y[sub].sum()} failures), "
                 f"one-sided permutation p={np.mean(np.array(null) >= within(y)):.2f}; "
                 + ", ".join(f"task {t} {roc_auc_score(y[task == t], full[task == t]):.2f}" for t in mixed))

    pilot = [episode_score(s) for ok, _, s in per_call("pi05pilot", "libero_10").values() if ok]
    tau = M.conformal_tau(pilot)
    lines.append(f"threshold from the {len(pilot)} LIBERO-Long successes at 10 steps (pi05pilot, conformal 90%): {tau:.4f}")
    lines.append(f"  LIBERO-90 above it: failures {(full[y] > tau).sum()}/{y.sum()}, successes {(full[~y] > tau).sum()}/{(~y).sum()}")
    lines.append(f"  median score: LIBERO-Long successes {np.median(pilot):.3f}, LIBERO-90 successes {np.median(full[~y]):.3f}, "
                 f"LIBERO-90 failures {np.median(full[y]):.3f}")
    (OUT / "followup.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def compare() -> None:
    """2 flow steps vs 10 on the screen's episodes. The reading rule is in NOTES.md (2026-11-11)."""
    ten, two = per_call(RUN, SUITE), per_call(RUN, SUITE, "pi05_s2")
    keys = sorted(set(ten) & set(two))
    calib = {c: [episode_score(s) for ok, _, s in per_call("pi05pilot", "libero_10", c).values() if ok]
             for c in ("pi05", "pi05_s2")}
    lines = [f"pi0.5 on LIBERO-90, 10 vs 2 flow steps: {len(keys)} matched episodes"]

    def above(eps, tau, failed):
        x = [episode_score(eps[k][2]) > tau for k in keys if eps[k][0] != failed]
        return sum(x), len(x)

    share = {}
    for name, c in (("inherited, from the 10-step", "pi05"), ("reset, from the 2-step", "pi05_s2")):
        tau = M.conformal_tau(calib[c])
        parts = []
        for steps, eps in ((10, ten), (2, two)):
            (nf, df), (ns, ds) = above(eps, tau, True), above(eps, tau, False)
            share[(c, steps)] = nf / df
            parts.append(f"{steps} steps: failures {nf}/{df} ({nf / df:.0%}), successes {ns}/{ds}")
        lines.append(f"threshold {name} LIBERO-Long successes (n={len(calib[c])}): {tau:.4f}")
        lines.append("  " + "; ".join(parts))
    for steps, eps in ((10, ten), (2, two)):
        y = [not eps[k][0] for k in keys]
        lines.append(f"pooled AUROC at {steps} steps: {roc_auc_score(y, [episode_score(eps[k][2]) for k in keys]):.2f}")
    a, b = share[("pi05", 10)], share[("pi05", 2)]
    lines.append(f"rule (collapse if the inherited threshold flags under half the 10-step share of failures, < {a / 2:.0%}): "
                 f"{'SmolVLA-like collapse' if b < a / 2 else 'partial or no loss'} ({a:.0%} -> {b:.0%})")
    (OUT / "compare.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("select", "report", "followup", "compare"))
    {"select": select, "report": report, "followup": followup, "compare": compare}[ap.parse_args().stage]()
