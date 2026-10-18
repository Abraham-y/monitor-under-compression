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
