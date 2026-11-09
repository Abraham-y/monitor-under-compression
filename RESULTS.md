# Results (run `main`, seed 0, LIBERO-Long, SmolVLA)

Sources:
- `results/main/table.md` and `results.json`: the pre-registered analysis (`analyze.py`).
- `results/main/robustness.md` and `.json`: review-driven robustness analysis (`robustness.py`, exploratory).
- `results/main/explore.json`: exploratory (`explore.py`).
- Figure 1: `paper/fig_main.pdf`. Paper: `paper/main.pdf`.

## Conditions

The test split is 100 episodes per condition: inits 40–49 × 10 tasks.

| condition | what it is | test success |
|---|---|---|
| ref | the checkpoint as shipped | 0.27 (0.32 over all 500) |
| int8 (`w8`) | RTN per-channel | 0.30 |
| 2 flow steps (`steps2`) | 2 Euler steps instead of 10; policy 0.81 → 0.32 s/call | 0.32 |
| int4-g32 (`w4g32`) | RTN groups of 32, all 521 layers; post hoc, exploratory | 0.05 |
| int4-coarse (`w4`) | pre-registered as "g128"; actually per-row for 62% of params (see NOTES) | 0.01 |

`w4nv` was stopped after review found the per-row fallback, and is not analysed.

Camera-ready check (run `schedule`, exploratory, success only): `ref_a10`, the original executing 10 actions per plan, succeeds in 0.33 of the test split against 0.27 (paired +0.06, 95% task bootstrap [−0.01, +0.14]). The replanning schedule does not explain the gap to the 0.56 public reproduction.

## Model vs schedule (exploratory; `shrink.py`, 100 matched test states)

| setting | success, 10 → 2 steps | action-consistency shrink, first 30 timesteps | failures above threshold, 10 → 2 steps |
|---|---|---|---|
| SmolVLA, replans every step | 0.27 → 0.32 | 0.42 | 21/73 → 0/68 |
| SmolVLA, 10 actions per plan | 0.33 → 0.44 (14 gained, 3 lost, p = 0.01) | 0.50 | 16/67 → 0/56 |
| π0.5, 10 actions per plan | 0.96 → 0.96 | 0.86 | 2/4 → 2/4 |

The thresholds are illustrative: conformal 90% on the same 10-step test successes, with no held-out split, so they differ from the paper's main results (36/73 → 0/68 for the first row). Only action consistency was scored at the other schedule and on π0.5. π0.5's four failures per variant are too few to judge detection.

## Main findings (inherited = ref-calibrated threshold, α = 0.1)

1. **Step reduction disarms the action-side monitors at unchanged success.**
   - act_cons TPR falls from 0.49 to **0.00** (0/68). McNemar 30 lost / 0 gained, p = 2e-9. ΔTPR −0.49 [−0.92, −0.25], bootstrapping tasks and tau together.
   - probe_exp TPR falls from 0.78 to 0.57 (14/0, p = 1e-4). Its median alarm moves from step 10 to step 124. This is threshold-sensitive (it vanishes at α = 0.05), so it is reported as directional.
   - probe_vlm (0.66 vs 0.67) and maha_vlm (0.32 vs 0.32) are unchanged. At step 0 the VLM state is bit-identical to ref on the matched tasks.
2. **Calibration, not information.**
   - act_cons disagreement is rescaled by a uniform 0.42 (IQR 0.40–0.44).
   - AUROC 0.77 → 0.72; oracle TPR at FAR ≤ 0.1 0.44 → 0.34; within-task AUROC 0.89 → 0.84.
   - Recalibrating on own successes recovers 0.26 TPR at FAR 0.03.
   - FAR *falls*: act_cons 0.15 → 0, probe_exp 0.11 → 0. A conformal check on successes passes while detection collapses.
3. **Quantization dose-response on the VLM probe.**

   | | int8 | int4-g32 | int4-coarse |
   |---|---|---|---|
   | step-0 score shift on identical inputs | −0.29 SD (100% toward "safe") | −2.30 SD | −4.58 SD |
   | probe_vlm TPR (ref 0.67) | 0.54 (13/3, p = 0.02) | 0.09 (39/0) | 0.00 (0/99) |

   - maha_vlm alarms on every int4 episode, successes included, so it detects the compression rather than the failures.

## Pre-registered hypotheses

- **H1** (w4 FAR > α): untestable. int4-coarse left 1 successful test episode.
- **H2** (w4 vs steps2 contrast): untestable, for the same reason.
- **H2b** (probe_exp |ΔFAR| larger under steps2 than under w8): supported (−0.11 vs −0.01).
- **H3** (w8 changes FAR and AUROC by ≤ 0.05): FAR holds for all 4 monitors. AUROC holds for 3/4; act_cons drops 0.08. Equivalence at ±0.05 cannot be established with n = 100.
- The detection changes were a pre-registered outcome, but no hypothesis predicted them.

## Robustness (exploratory; see robustness.md)

- **Episode length:** counting only alarms before step 200 lowers ref TPRs (probe_vlm 0.48, act_cons 0.27). Both zeros stay zero.
- **Task identity:** within-task ref AUROC is 0.96 (probe_vlm) and 0.98 (probe_exp). Excluding the two never-succeeding tasks changes no conclusion.
- **Matched step-0 inputs:** tasks 3, 8 and 9 have cuDNN-autotuning differences in ref's step-0 state, so they are excluded from same-input comparisons. The headline numbers hold without them.

## Relation to concurrent work

The concurrent work is an anonymous NeurIPS 2026 workshop submission (OpenReview atOkXTzFCL). It replays fp32 SmolVLA observations open loop through per-channel int4/int8 weights, uses probes only, and calibrates on failures. It finds coverage shifts at unchanged AUROC and repairs them by ridge realignment.

This work differs in four ways:
- the compressed policies run closed loop;
- it includes step reduction;
- it uses non-probe monitors;
- it calibrates on successes (FAR-bounding), which turns the shift into silent misses.

Both works agree on the rank-preserving, threshold-breaking mechanism.
