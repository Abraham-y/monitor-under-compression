"""Does fewer flow steps shrink action-consistency scores? Paired, label-free (exploratory).

    python shrink.py --run main --a ref --b steps2 --offset 1                  # SmolVLA, replans every step
    python shrink.py --run schedule --a ref_a10 --b steps2_a10 --offset 10     # SmolVLA, 10 actions/plan
    python shrink.py --run pi05pilot --a pi05 --b pi05_s2 --offset 10          # pi0.5, 10 actions/plan
    # -> results/<run>/shrink_<a>_vs_<b>.txt

Compares condition b (2 flow steps) with a (10) on the test split's matched start states.
offset is the number of actions executed per plan, so a policy call happens every `offset`
timesteps and successive plans overlap by 50 - offset actions. The warm-up (10 timesteps)
and the early window (first 30 timesteps) are converted to calls, so every setting is scored
over the same stretch of the episode. The paper's label-free check needs only successful
runs; detection rates use an illustrative threshold set on the same a-condition successes
(no held-out split).
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

import monitors as M

ROOT = Path(__file__).resolve().parent
WARMUP_T, EARLY_T = M.BURN_IN, 30  # timesteps


def load(run: str, cond: str, offset: int) -> dict:
    warm, early = -(-WARMUP_T // offset), -(-EARLY_T // offset)
    eps = {}
    for f in sorted((ROOT / "data" / run / cond / "libero_10").glob("*_init40-49_s0.npz")):
        z = np.load(f, allow_pickle=False)
        meta = ast.literal_eval(str(z["meta"]))
        for b, init in enumerate(z["init_ids"]):
            L = int(z["length"][b])
            n = -(-L // offset)  # calls made before this env finished
            s = M.chunk_disagreement(z["chunk"][b, :n], offset)
            S = M.running_mean(s)
            eps[(int(meta["task_id"]), int(init))] = {
                "success": bool(z["success"][b]),
                "score": float(S[warm:].max()) if len(S) > warm else float(S[-1]),
                "early": float(s[:early].mean()),
            }
    return eps


def q(x) -> str:
    return f"median {np.median(x):.2f}, middle half {np.percentile(x, 25):.2f}-{np.percentile(x, 75):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pi05pilot")
    ap.add_argument("--a", default="pi05")
    ap.add_argument("--b", default="pi05_s2")
    ap.add_argument("--offset", type=int, default=10)
    args = ap.parse_args()
    a, b = load(args.run, args.a, args.offset), load(args.run, args.b, args.offset)
    keys = sorted(set(a) & set(b))
    lines = [f"{args.a} vs {args.b} (run {args.run}), {len(keys)} matched states, offset {args.offset}"]
    sa = np.array([a[k]["success"] for k in keys])
    sb = np.array([b[k]["success"] for k in keys])
    gain, loss = int((sb & ~sa).sum()), int((sa & ~sb).sum())
    p = binomtest(gain, gain + loss, 0.5).pvalue if gain + loss else 1.0
    lines.append(f"success: {args.a} {sa.mean():.2f}, {args.b} {sb.mean():.2f} "
                 f"({args.b}-only {gain}, {args.a}-only {loss}, McNemar p={p:.2f})")
    both = [k for k in keys if a[k]["success"] and b[k]["success"]]
    lines.append(f"early disagreement ratio b/a, first {EARLY_T} timesteps: all pairs "
                 f"{q([b[k]['early'] / a[k]['early'] for k in keys])}")
    if both:
        lines.append(f"  runs succeeding in both ({len(both)}): {q([b[k]['early'] / a[k]['early'] for k in both])}")
        lines.append(f"episode-score ratio b/a, both succeed: {q([b[k]['score'] / a[k]['score'] for k in both])}, "
                     f"Wilcoxon p={wilcoxon([b[k]['score'] for k in both], [a[k]['score'] for k in both]).pvalue:.1e}")
    succ_a = [a[k]["score"] for k in keys if a[k]["success"]]
    succ_b = [b[k]["score"] for k in keys if b[k]["success"]]
    tau = M.conformal_tau(succ_a)
    fa, fb = [k for k in keys if not a[k]["success"]], [k for k in keys if not b[k]["success"]]
    lines.append(f"illustrative threshold (conformal 90% on the {len(succ_a)} {args.a} successes): {tau:.4f}")
    lines.append(f"  successes above it: {args.a} {sum(s > tau for s in succ_a)}/{len(succ_a)}, "
                 f"{args.b} {sum(s > tau for s in succ_b)}/{len(succ_b)}")
    lines.append(f"  failures above it: {args.a} {sum(a[k]['score'] > tau for k in fa)}/{len(fa)}, "
                 f"{args.b} {sum(b[k]['score'] > tau for k in fb)}/{len(fb)}")
    out = ROOT / "results" / args.run / f"shrink_{args.a}_vs_{args.b}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
