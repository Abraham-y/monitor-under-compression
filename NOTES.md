# NOTES

## 2026-10-17

- Repo created. lerobot 0.6.1, checkpoint `HuggingFaceVLA/smolvla_libero`
  (chunk 50, n_action_steps 1, 10 Euler steps, SmolVLM2-500M backbone, expert width 0.5).
- Novelty sweep (web + arXiv keyword, 2026-10-13): no paper evaluates a VLA failure
  monitor on a compressed version of its policy. Closest: SAFECAST (calibration under
  input shift), VLAQuantBench (PTQ success only), arXiv 2606.03002 (SAE features
  degrade under LLM quantization while perplexity holds).
- Synthetic tests for monitors.py pass (tests/test_monitors.py).

## Smoke runs (2026-10-17, L4, plumbing and success only)

- LIBERO asks for a dataset path on first import; the image answers "n" at build time.
- Timing: ~1.2-1.9 s per env step; policy 0.5-0.8 s per call and nearly flat in batch
  size (2 vs 10 envs: 0.50 vs 0.60 s), so batches of 10 cost about the same as 2.
  `steps2` cuts policy time to 0.24 s per call (3.2x).
- Quantization check: w8 max 205 distinct values per row (limit 255), w4 exactly 16 per
  group; 521 Linear layers, 508M of 556M Linear params.
- libero_spatial task 0, success by init state:
  - rollout.py, batch 10: 6/10 (inits 0,4,5,6,7,8)
  - lerobot-eval, batch 10: 6/10 (inits 0,3,6,7,8,9)
  - lerobot-eval, batch 1: 1/3 (init 0)
  - rollout.py batch 4: ref 1/4, w8 2/4, w4 3/4, steps2 2/4; init 0 succeeds in all.
  Same rate as the official evaluator, so the pipeline is not the cause of the gap to the
  paper. Versions: lerobot 0.6.1, mujoco 3.8.1, robosuite 1.4.0, torch 2.11.0.
- libero_10 task 0, ref, 2 episodes: 0/2 (520 steps each).

## Main collection (launched 2026-10-18 12:12 PDT, after the user approved ~$25-35)

- 40 jobs (ref x10 tasks with 5 batches each; w8/w4/steps2 x10 tasks with 2 batches), L4, 10 at a time.
- ~7.7-8.3 min per batch of 10 libero_10 episodes; policy 0.61 s/call.
- Post-freeze code changes, none touching pre-registered analyses:
  - monitors.py: MAHA_PCS readable from env MUC_MAHA_PCS (default 64 unchanged), for tests.
  - analyze.py: exploratory `t0_shift` (labelled as such): step 0 sees the same observation
    in every condition, so h[0] differences measure representation shift for an identical
    input; also a sanity check that steps2 leaves h_vlm[0] unchanged.
  - figures.py, tests/test_pipeline.py added.
- 12:55: all 30 compressed (condition, task) inputs of the first launch failed at
  `vol.reload()` ("open files preventing the operation: hf/xet/logs/..."), before any
  rollout. Ref inputs unaffected. reload is now best-effort; compressed conditions
  relaunched as a separate app (`--conditions w8,w4,steps2`). No data lost or duplicated.

## 2026-10-18 afternoon

- Pre-registered analysis done (RESULTS.md). w4 collapsed task success (0.01), so H1/H2
  were not testable as specified.
- Post hoc, approved by the user (~$5-8): `w4nv` = w4 with the vision encoder left in full
  precision, inits 30-49 seed 0, to look for an int4 regime that keeps task success.
  Exploratory; reported separately from the pre-registered conditions.
- Second seed: not run (user agreed to the recommendation to skip it); the headline
  effects are on detection with 68-99 failures per condition.
- Four adversarial review agents launched (analysis code, rollout hooks, hostile reviewer
  + novelty, data robustness). Read-only; findings to be logged here.

## Adversarial review (2026-10-18 evening): findings and actions

Four read-only review agents. All four reproduced the headline numbers.

1. **Analysis code**: no bug changes a headline.
   - **Tasks 3, 8, 9 step-0 mismatch.** Their ref step-0 h_vlm differs from steps2/w8, which match each other.
     - Cause per the rollout reviewer: per-process cuDNN autotuning (`cudnn.benchmark=True`) of SigLIP's conv.
     - The other 7 tasks are bit-identical.
     - Action: same-input analyses use the matched tasks only; "noise floor 0.16" retracted.
   - **w4 AUROCs rest on n_success=1.** Now reported as "—" in robustness.md.
   - **w8 claim.** Now "no detectable change in FAR", not equivalence.
   - **Episode length.** Added detection with alarms counted only before step 200.
   - **Task identity.** Added within-task AUROC.
   - **Noise reuse.** The same torch seed per batch reuses noise across batches. Noted; effect negligible.
2. **Rollout hooks**: the hooks, alignment, done/success handling and steps2 are all correct.
   - **HIGH: w4 fell back to one group per row** for in_features not divisible by 128. That is 384/521 layers and 62% of params, including every text-decoder and expert projection.
     - Action: w4 relabelled "int4-coarse".
     - w4nv (built on the same fallback) stopped after 10/20 batches (1/100 success, not analysed).
     - Corrected int4 `w4g32` (groups of 32, all layers) run post hoc.
   - **Lower-severity notes:** h_vlm pooled in bf16; the sync envs are not wrapped in FreezeAfterEpisodeEnd (safe, because we truncate at length); w4nv still quantized the connector.
3. **Hostile reviewer**: score 2/5 as drafted, ~3 with fixes.
   - **Novelty.** Found concurrent work (anonymous NeurIPS 2026 workshop submission, OpenReview atOkXTzFCL; the user pasted the full text): open-loop paired activations, probe-only, simulated per-channel int4/int8, coverage of failures; AUC unchanged while coverage shifts; ridge realignment repair.
   - **Our distinct contributions:** closed loop, step reduction, non-probe monitors, success-calibrated (FAR-bounding) thresholds that hide detection loss.
   - **Action:** reframed the paper around step reduction ("Recalibrate After You Accelerate"); cited the concurrent work, Duan 2606.15980 and CARE 2610.08917 (both verified on arXiv).
4. **Robustness**:
   - **F1 (act_cons 0/68 under steps2)** survives every check. It is a uniform 0.42x rescaling, i.e. a calibration failure: oracle TPR 0.44 -> 0.34, AUROC 0.77 -> 0.72.
   - **F2 (probe_exp)** is directional and threshold-sensitive.
   - **F3 (probe_vlm 0/99 under int4-coarse)** survives.
     - Mahalanobis alarms on everything, successes included.
     - The shift is large and nearly orthogonal to the failure direction; its projection on the probe weights flips the score.
   - **F4 (w8)** weakened: probe_vlm TPR -0.13 (McNemar 13/3, p=0.02).

New analysis: `robustness.py` -> results/main/robustness.{json,md}.
