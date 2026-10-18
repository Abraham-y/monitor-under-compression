# Results (run `main`, seed 0, LIBERO-Long, SmolVLA)

Numbers: `results/main/results.json`, `results/main/table.md` (pre-registered analysis,
`analyze.py`), `results/main/explore.json` (exploratory, `explore.py`). Figures:
`paper/fig_main.pdf`, `paper/fig_scores.pdf`.

## Task success (test split, 100 episodes each)

| ref | int8 W | int4 W | 2 flow steps |
|---|---|---|---|
| 0.27 | 0.30 | **0.01** | 0.32 |

Reference over all 500 episodes: 0.32. Tasks 0 and 4 never succeed (0/50 each).
int4 W falls below half of ref, so Gate 2 flags it as outside the "success unchanged"
regime.

## Monitors on the reference policy (the control)

| monitor | AUROC | TPR | FAR (α = 0.10) |
|---|---|---|---|
| probe_vlm | 0.95 | 0.67 | 0.04 |
| probe_exp | 0.95 | 0.78 | 0.11 |
| maha_vlm | 0.85 | 0.32 | 0.07 |
| act_cons | 0.77 | 0.49 | 0.15 |

All four FARs are consistent with the conformal guarantee.

## Pre-registered hypotheses

- **H1** (w4 FAR > α for a hidden-state monitor): **not testable as specified.** int4
  left 1 successful test episode. maha_vlm alarmed on it (FAR 1/1, p = 0.10).
- **H2** (|ΔFAR| smaller under steps2 than w4 for VLM monitors, reversed for act_cons):
  **not evaluable**, because w4's FAR rests on n = 1. Under steps2 the VLM-side monitors'
  FAR is unchanged (maha −0.01, probe_vlm −0.04). act_cons FAR falls by 0.15 (95% CI
  −0.30 to −0.04).
- **H2b** (probe_exp |ΔFAR(steps2)| > |ΔFAR(w8)|): **supported, directionally.** The
  values are −0.11 vs −0.01.
- **H3** (w8 changes FAR by ≤ 0.05 and AUROC by ≤ 0.05): FAR holds for all four
  monitors. AUROC holds for 3 of 4; act_cons drops 0.08 (0.77 → 0.69).

## What actually moved: detection, not false alarms

TPR is a pre-registered outcome but no hypothesis predicted these changes. All values use
the reference-calibrated threshold.

- **2 flow steps** is a 3.2x faster policy with unchanged success:
  - act_cons TPR falls from 0.49 to **0.00** [0.00, 0.05] across 68 failures. Every
    failed episode's score sits below τ: the median is 0.13 against τ = 0.37.
  - probe_exp TPR falls from 0.78 to 0.57, and its median alarm comes later (2% → 24% of
    the episode).
  - The VLM-state monitors are unchanged: probe_vlm 0.66 vs 0.67, maha 0.32 vs 0.32.
  - Mechanism: the VLM prefix is untouched (exploratory t0 shift 0.16, the cross-process
    noise floor), while the expert state moves by 2.4 ref-spreads and the action chunks
    get smoother.
- **int4 W** (policy collapses):
  - probe_vlm, the best monitor on ref (AUROC 0.95), raises **0 alarms on 99 failed
    episodes** (TPR 0.00 [0.00, 0.04]). Its failure-episode scores have median −8.9
    against τ = 2.39. On identical step-0 inputs its score moves −4.9 ref-SD toward
    "success".
  - maha_vlm alarms on everything (TPR 1.00), which is correct here.
  - probe_exp TPR is 0.92 and act_cons is 0.68.
- **int8 W**: everything stays within noise except a probe_vlm TPR dip (0.67 → 0.54) and
  the act_cons AUROC drop noted above.
- **Recalibrating on the compressed policy's own successes** partly restores the
  monitors under steps2 (act_cons TPR 0.26 at FAR 0.03; probe_exp 0.66 at 0.06). It is
  impossible under w4, which has no successes to calibrate on.

## Exploratory

- **Excluding the two never-succeeding tasks** (so the probes can't just recognise task
  identity) changes none of the conclusions:
  - probe AUROCs on ref are 0.93.
  - probe_vlm TPR under w4 is still 0.
  - act_cons TPR under steps2 is still 0.
- **t0 representation shift** (identical step-0 input, relative to the ref spread across
  init states):

  | condition | h_vlm | h_exp |
  |---|---|---|
  | w8 | 0.36 | 0.20 |
  | w4 | 2.70 | 1.82 |
  | steps2 | 0.16 | 2.39 |

  steps2's h_vlm shift is not exactly zero. It is probably kernel nondeterminism across
  processes; we treat it as the noise floor.
