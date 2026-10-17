# Pre-registration

**Status: FROZEN 2026-10-17, before the main collection.** The smoke runs (listed in
NOTES.md) only checked plumbing, quantization, timing and task success. No monitor was fit
and no monitor score was computed on real rollouts before the freeze. Any later change goes
in the change log at the bottom, with its date and reason.

## Question

A VLA deployed on a robot is usually compressed to meet latency. It also usually has a
runtime failure monitor, and that monitor's alarm threshold is calibrated on rollouts
of the full-precision policy. Does the monitor still work on the compressed policy?
Two specific things could go wrong:

1. **Calibration.** The conformal threshold promises a false-alarm rate of at most α on
   successful episodes. That guarantee assumes the test episodes come from the same
   process as the calibration episodes. A compressed policy is a different process.
2. **Separability.** The monitor might stop ranking failures above successes.

## Setup

- **Policy:** `HuggingFaceVLA/smolvla_libero` (SmolVLA, ~450M) run through LeRobot 0.6.1 with the
  checkpoint's own settings: 50-step chunk, replanning every step, 10 Euler steps.
- **Benchmark:** LIBERO-Long (`libero_10`), all 10 tasks, hard resets, fixed init states.
  It is the suite where SmolVLA fails most often, which gives the monitors failures
  to detect.
- **Conditions:**
  - `ref`: the checkpoint as shipped.
  - `w8`: RTN weight-only int8, per output channel, symmetric.
  - `w4`: RTN weight-only int4, groups of 128 along the input dimension, asymmetric.
  - `steps2`: 2 Euler steps instead of 10, with weights untouched.

  The quantizers cover every `nn.Linear` inside the VLM and the action expert (vision
  encoder, text decoder, expert). The policy's small projection heads stay in full precision. The
  quantization is simulated: weights are quantized and then dequantized in place. That
  reproduces the numerics but not the latency. Before any rollout, a check confirms the
  quantized weights take at most 2^bits distinct values per group.
- **Splits, by init state (identical for every task):**
  - Train is init 0–29, `ref` only.
  - Calibration is init 30–39, every condition.
  - Test is init 40–49, every condition.
  - Seed 0 throughout.
  - So `ref` has 500 episodes and each compressed condition has 200.
- **Logged per step:** the VLM prefix state (mean-pooled after the final norm), the
  action-expert state at the last denoising step (mean-pooled over the chunk), and the
  normalized action chunk.

## Monitors

All four are fixed in `monitors.py`, which is committed with this file.

| name | input | needs failure labels? | family |
|---|---|---|---|
| `probe_vlm` | logistic probe on the VLM prefix state | yes | SAFE |
| `probe_exp` | logistic probe on the action-expert state | yes | SAFE |
| `maha_vlm` | Mahalanobis distance to successful training steps (64 PCs, Ledoit-Wolf) | no | VLA-FAIL LLMD, FIPER |
| `act_cons` | disagreement of consecutive chunks on their 49 shared steps | no training | Sentinel, VLA-FAIL ACC |

- **Episode score:** S_t is the running mean of the per-step score.
- **Alarm:** fires at the first t ≥ 10 with S_t > τ.
- **Threshold:** τ is the conformal (1−α) quantile, with α = 0.10, of max_{t≥10} S_t over
  **successful `ref` calibration episodes**. This is the "deployed" threshold.
- **Probe regularization:** each probe's C is chosen by 5-fold grouped CV (grouped by
  episode) on the train split.

## Outcomes, all on test episodes

| outcome | definition |
|---|---|
| FAR | fraction of successful episodes that raise an alarm, with the `ref` threshold |
| TPR | fraction of failed episodes that raise an alarm, with the `ref` threshold |
| AUROC | AUROC of max_{t≥10} S_t, failure versus success |
| recalibrated FAR and TPR | the threshold recomputed from the condition's own successful calibration episodes |
| task success | per condition, with Wilson 95% CIs |

## Hypotheses

- **H1, calibration drift (primary):**
  - For at least one hidden-state monitor (`probe_vlm`, `probe_exp`, `maha_vlm`), FAR under
    `w4` exceeds the nominal α = 0.10.
  - Test: one-sided exact binomial, p < 0.05.
  - `ref`'s own FAR is reported as the control. It should be at or below 0.10, up to
    sampling noise.
- **H2, monitor type × compression type (primary):**
  - `steps2` leaves the VLM prefix computation untouched for a given observation.
    Its hidden-state shift therefore comes only from the policy visiting different
    states.
  - Prediction: |FAR(`steps2`) − FAR(`ref`)| < |FAR(`w4`) − FAR(`ref`)| for `maha_vlm` and
    `probe_vlm`.
  - Prediction: the reverse ordering holds for `act_cons`.
  - Reported with bootstrap CIs (episodes resampled within condition, 10,000 draws).
    The claim is directional, with no significance threshold.
- **H2b (secondary):**
  - Under `steps2`, the action-expert state is read at the last of 2 Euler steps instead
    of the last of 10, so its input timestep changes.
  - Prediction: |FAR(`steps2`) − FAR(`ref`)| > |FAR(`w8`) − FAR(`ref`)| for `probe_exp`.
- **H3 (secondary):**
  - `w8` changes no monitor's FAR by more than 0.05 from `ref`.
  - |ΔAUROC| ≤ 0.05 under `w8` for every monitor.
- **Exploratory:**
  - Recalibration as a fix.
  - Alarm timing.
  - Whether success-rate changes explain FAR changes.

## Decision rules

- **Gate 1 (pipeline):** `rollout.py` must reproduce LeRobot's own `lerobot-eval` on the
  same task and init states. It passed before the freeze (NOTES.md). Neither tool
  reaches the paper's 71% on this setup: community reproductions of this checkpoint
  report 56% on LIBERO-Long (lerobot #3264). So the paper's number is not a gate. The
  reference success rate is reported as measured.
- **Gate 2:** if a compressed condition's task success falls below half of `ref`'s, it is
  still run and reported. It is flagged as outside the "success unchanged" regime the
  question targets.
- **Small samples:** if a condition has fewer than 15 test failures, AUROC and TPR are
  reported with a warning. FAR is computed regardless.
- **Null results:** reported as nulls.

## Change log

- 2026-10-17: draft written during the smoke runs.
- 2026-10-17, before the freeze: Gate 1 changed from "within 15 points of the paper's
  71%" to "matches `lerobot-eval`". The reason is that the official tool itself scores
  6/10 on LIBERO-Spatial task 0, where the paper implies about 90%. This is a known
  reproduction gap (lerobot #3264, #4614) and not a pipeline bug. H2b added. Both
  changes were made before any libero_10 data beyond the 2-episode plumbing run.
