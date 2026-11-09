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

## 2026-10-19: citation audit and abstract rewrite (user request)

- Downloaded all 14 cited arXiv papers and the LeRobot issue, then checked every attributed claim: references/CITATION_CHECK.md.
- Fixed:
  - the Duan sentence (overstated);
  - the LIBERO "50 initial states" attribution (it comes from the benchmark code, not the paper);
  - "community reproductions" became "a public reproduction" (lerobot#3264, cited by URL).
- Added: SAFE, FIPER and Sentinel all calibrate on successful rollouts, so the one-sided-calibration blind spot is standard practice.
- The concurrent paper was checked against the user-pasted full text. It is not downloaded, because OpenReview's browser check blocks it.
- Rewrote the abstract for newcomers. The main text still fits 4 pages exactly.
- Clarity pass (user OK'd):
  - Rewrote the intro, setup and results in plain language, defining each term at first use.
  - Dropped the separate quantization table and folded its numbers into the text.
  - Every number was re-checked against results/main. The previous version is tagged `pre-clarity-pass`.

## 2026-10-19: second adversarial review of the plain-language draft (4 agents)

**Reviewer 2: 3/5, about 4/5 with text fixes. Compliance passed** (anonymous PDF, 4 pages, no overfull boxes).

**Applied, after re-verifying each finding myself** (scratchpad/verify_review.py):
- **Title:** singular ("Disables a VLA Failure Monitor"); "Re-test After You Accelerate".
- **Expert probe:** reported as suggestive. The task CI is [-0.41, +0.09], and the drop vanishes at alpha=0.05 (.33 vs .32, recomputed).
- **Action consistency** stays 0 at alpha=0.05. On matched tasks it is 21/0 (p=1e-6).
- **int8 VLM-probe drop:** not significant on the matched tasks (8/2, p=0.11). Now stated.
- **New label-free check:** act_cons scores on successful runs shrink (median 0.27 -> 0.11; success-pair ratio 0.43). Added as the cheap re-test in the abstract and takeaways.
- **Pre-registration accounting corrected:** H1 and H2 untestable; H2b held; H3 failed (act_cons AUROC -0.08). Exploratory analyses now labelled.
- **Overclaims fixed:** "just as successful", "unchanged", "constant", "no learned parameters", "most practical", "brings back".
- **Figure:** re-exported at its printed width (5.5 in, 7 pt+ text). The caption explains the intervals and that both int4 variants break the policy.
- **Footer and PDF subject** now name the EEAI workshop.
- **Citation fixes:** see references/CITATION_CHECK.md, second audit.

## 2026-10-19: third (final) review round (fact delta, citation delta, proofread)

- **Nothing blocking:** anonymity, page limit and number consistency all pass.
- **Fixes:**
  - The matched-task footnote is narrowed to "the detection results hold except int8's" (the original act_cons already had 0/17 false alarms on those tasks).
  - Bootstrap CIs updated to the current robustness.json: [-0.92, -0.25] and [-0.41, +0.08].
  - Re-tuning is marked as pre-registered (it is in the PREREG outcomes table).
  - H1/H2 are now "uninformative" rather than "could not be tested".
  - "0.43 for runs that succeed in both".
  - RESCUE's miss is on held-out tasks.
  - Action consistency is "after" VLA-FAIL, since VLA-FAIL's version is normalized and position-only.
  - "the 521 linear layers in the backbone and action expert".
  - Terminology: "2-step policy" and "initial state" everywhere.
  - The abstract reads "27% before, 32% after" and calls LIBERO-Long a benchmark.
  - Figure tick label "int4-coarse"; pdftitle set.

## 2026-11-07: accepted; camera-ready

**Decision:** accepted 2026-11-05. One review (7/10, confidence 3) plus a meta-review; the text is in SUBMISSION.md. Asks: validate the rollout schedule, keep observed success apart from non-inferiority, scope claims to our monitor implementations, narrow "threshold is what breaks". Text-only revision (user's choice); no new Modal runs.

**Changes to `paper/main.tex`:**
- De-anonymized: `[final]` option, author Abraham Yeung, Stanford, repo link in the intro. The footer names the workshop.
- "No loss of success" became "no observed drop". New paired numbers: 10 states gained, 5 lost, 95% task-bootstrap interval −1 to +12 points (`success_diff.py`, results/main/success_diff.txt).
- "Simplified versions of" in the abstract; Limitations says the monitors simplify SAFE/VLA-FAIL/Sentinel rather than reproduce them.
- Results heading is now "Most of the loss is the inherited threshold; some is ranking". The best hindsight threshold catches 0.44 → 0.34, so about 10 points go to ranking and 34 to the threshold.
- Setup states the 1-action replanning schedule as shipped. The footnote adds the released checkpoint's expert width 0.5 (the paper uses 0.75; the HF config was checked) and MuJoCo ≥3.4 start-scene drift (lerobot#4390, 152/500 LIBERO-10 states; we ran 3.8.1).
- Related-work sweep (subagent; every cited paper re-verified on arXiv or a venue page): added coupling (2610.32102), onestep (2610.30735, cited as arXiv because the abs page names no venue), doula (NeurIPS 2026 E&D), actprobe (2606.08508) and faildetect (RSS 2025). rankpreserving is now Bysani, RoboPAD @ NeurIPS 2026; quantvla is now CVPR 2026.
- The "we did not test why 2-step plans agree more" limitation is replaced by the coupling citation (the shrink factor fits 0.43–0.51 for π0/π0.5).
- Trimmed to fit 4 pages. No overfull boxes.

**Not done:**
- int8 vs concurrent work: they say int8 leaves the VLM text stream untouched; we see a 0.29 SD probe shift. We also quantize the vision encoder. Unverified, and left out of the paper.

## 2026-11-07 (later): replanning-schedule check, sample media, acknowledgment

**Schedule check (the reviewer's main ask; user approved, about $2–4):**
- New condition `ref_a10`: the shipped weights with `--policy.n_action_steps=10`. Run as plan `schedule` on the test split (inits 40–49, seed 0), kept out of run `main`.
- rollout.py now counts policy calls as ceil(T / n_action_steps) instead of assuming one per step.
- Smoke on task 1 gave 52 calls for 520 steps.
- Result: 33/100 against ref's 27/100 on the same states. +6 points, 11 gained / 5 lost, 95% task-bootstrap [−1, +14], McNemar p=0.21 (`success_diff.py`).
- Even the upper bound (~41%) is far below the 56% public reproduction, so the schedule does not explain the gap. Stated in the setup paragraph.
- Monitors were not evaluated under ref_a10, so Limitations still says "one replanning schedule".

**Sample media (`media.py`, outputs in `media/`):**
- Frames come from `modal_app.replay`, a CPU-only stage that replays each episode's recorded actions from the same init state and env seed.
- All 6 replays matched the recorded success and length; both successes ended on exactly the recorded step (264 and 240). So the frames are the episodes the paper scored.
- Clips:
  - fail_task2_init42: ref alarms at step 100; 2-step peaks at 0.31× threshold.
  - fail_task4_init45: ref alarms at step 145; 2-step peaks at 0.42×.
  - success_task2_init46: neither alarms; peaks 0.82× and 0.31×.
- Avoided tasks 0, 1 and 7 (MuJoCo ≥3.4 start-scene drift) and 3, 8 and 9 (step-0 feature mismatch).
- Each clip has an MP4 (1280×720, 30 fps), a GIF (640 px, 2× speed), a still and a filmstrip (PNG and PDF).

**Paper:** added `\acknowledgments{We thank Modal for compute.}`. It prints after the main text, on page 5 before the references.

## 2026-11-07 (night): pi0.5 pilot (post-workshop, exploratory; user approved; about $3–4)

**Setup:**
- `lerobot/pi05_libero_finetuned_v044` on LeRobot 0.6.1 (the `pi` extra, added as its own image layer). Compilation off; the checkpoint runs in bf16.
- The checkpoint ships `n_action_steps=50`, which leaves successive plans no overlap. So the pilot executes 10 actions per plan (40-action overlap), and disagreement is computed at offset 10 (`monitors.chunk_disagreement(..., offset)`; offset=1 is unchanged).
- Conditions `pi05` (10 flow steps) and `pi05_s2` (2 steps), test inits 40–49, seed 0, 100 matched states.
- Chunks only; h_vlm/h_exp are placeholders. Analysis: `shrink.py` -> results/pi05pilot/shrink_pi05_vs_pi05_s2.txt.
- The gated PaliGemma tokenizer loaded with the Modal HF secret.

**Results:**
- Success 0.96 vs 0.96 (1 gained, 1 lost).
- Policy time per call (batch 10, L4): 1.84 s vs 1.52 s. Only about 1.2x, because the 2B backbone dominates; SmolVLA gets 2.5x.
- On the 95 states where both succeed, 2-step disagreement over the first 30 timesteps is 0.86x (middle half 0.76–0.94). Episode score is 0.88x (0.75–1.05), Wilcoxon p=8e-4. SmolVLA was 0.42x.
- Illustrative threshold (conformal 90% on the same 10-step successes): 4/96 2-step successes exceed it.
- Only 4 failures per condition, so no detection estimate.

**Reading:**
- The direction replicates: scores shrink and false alarms would drop. The size does not.
- Confound: the schedule differs (offset 10 here, offset 1 for SmolVLA). Comparing plans 10 steps apart may be dominated by real replanning rather than sampling spread.
- Cheapest disambiguation: SmolVLA 2-step at 10 actions per plan (`steps2_a10`, about 10 batches). Its counterpart `ref_a10` already exists.
- A full pi0.5 detection run needs many more failures than 4%/state gives, e.g. harder or perturbed variants.

## 2026-11-08: model vs schedule (SmolVLA 2 steps at 10 actions per plan; user approved; about $2)

`steps2_a10` (run `schedule`, test inits 40–49, seed 0) against `ref_a10`, offset 10. All three comparisons use `shrink.py`:

| setting | success, 10 -> 2 steps | early shrink (all pairs) | episode-score ratio, both succeed | failures above threshold, 10 -> 2 steps |
|---|---|---|---|---|
| SmolVLA, replans every step | .27 -> .32 | 0.42 | 0.38 | 21/73 -> 0/68 |
| SmolVLA, 10 actions/plan | .33 -> .44 (14 gained / 3 lost, p=.01) | 0.50 (both succeed: 0.44) | 0.44 | 16/67 -> 0/56 |
| pi0.5, 10 actions/plan | .96 -> .96 | 0.86 | 0.88 | 2/4 -> 2/4 |

The thresholds here are illustrative: conformal 90% on the same 10-step successes, no held-out split. The paper's own threshold for the first row (held-out calibration) gives 36/73 -> 0/68.

**Reading:**
- It is the model, not the schedule. SmolVLA's shrink stays near 0.4–0.5 at 10 actions per plan, and the detection collapse repeats there. pi0.5's shrink is far smaller at the same schedule.
- Untested candidate cause: SmolVLA's flow is the most curved of the models in Sujit et al. (1- vs 10-step endpoint error 29%).
- At 10 actions per plan, 2 steps significantly raises SmolVLA's success. This fits Yang et al.'s finding that coarse integration helps under longer open-loop commitment.
- shrink.py gives 0.42 for SmolVLA runs succeeding in both, and the paper's earlier 0.43 used inits 30–49 (39 pairs) rather than the test set. On the test set both windows give 0.42 (22 pairs); the paper now says so.

**Correction to the camera-ready (my error):** I cited Yang et al.'s 0.43–0.51 next to our 0.42 as if they agreed. Their ratio is a cloud radius (mean L2 distance to the sample mean); our score is a mean squared difference between successive plans, so the numbers are not comparable. In root-mean-square terms ours are about 0.65 (SmolVLA) and 0.93 (pi0.5). pi0.5 also shows that shrinking within-observation spread need not shrink this score much. The sentence now cites them qualitatively ("consistent with"), and we still have not tested the mechanism. The public PDF at ef21c7f still has the old sentence until the next push.

**Added to the camera-ready on 2026-11-08 (user approved):** a Results sentence on the 10-action collapse and a Limitations sentence on the pi0.5 pilot, plus a citation for pi0.5 (arXiv 2504.16054). To hold 4 pages I cut: "Expect a monitor to break where its input changes" (Takeaways; pi0.5 weakens it anyway), the sentence repeating that the threshold hides the loss, "we lose only monitoring", and "repaired by realigning its activations". Abstract and TL;DR are unchanged.
