# EEAI @ CoRL 2026 submission

## Status: accepted (2026-11-05), camera-ready in progress

- **Decision:** Accept (submission #45). One review, 7/10 ("Good paper, accept"), confidence 3.
- **Workshop:** Thursday 2026-11-12, 14:00–18:00 CST, Hilton Austin Room 408. Every accepted paper gets a poster (session at 15:35). About five also get a 5-minute lightning talk.
- **Camera-ready:** the workshop site gives no instructions or deadline. We upload the de-anonymized PDF as a revision on OpenReview. `paper/main.pdf` is the camera-ready version; `paper/eeai_submission.pdf` is the version as submitted.
- **Reviewer asks and responses:**
  - **Validate the rollout schedule (the baseline gap).** Tested.
    - The setup states the 1-action replanning schedule the checkpoint ships with.
    - Executing 10 actions per plan (`ref_a10`, same 100 test states) gives 33% vs 27%: +6 points, 95% [−1, +14]. So the schedule does not explain the gap to the 56% reproduction.
    - Also added: an independent evaluation at ~35%, the checkpoint-width difference and MuJoCo ≥3.4 drift.
    - The full monitor evaluation uses the shipped schedule. Action consistency alone was also checked at 10 actions per plan (`steps2_a10`, exploratory): shrink 0.50x, failures above an illustrative threshold 16/67 → 0/56, success 33% → 44% (14 gained, 3 lost, p = 0.01).
    - A π0.5 pilot (100 test states, 10 actions per plan) shrinks only 0.86x, so the paper's Limitations say the effect's size depends on the policy.
  - **Keep observed success separate from non-inferiority.** "No loss of success" became "no observed drop", with a paired 95% task-bootstrap interval of −1 to +12 points (10 initial states gained, 5 lost).
  - **Claims concern these monitor implementations, not the full SAFE/VLA-FAIL/Sentinel.** Stated in the abstract ("simplified versions") and in Limitations.
  - **Narrow "threshold is what breaks".** The paragraph is now "Most of the loss is the inherited threshold; some is ranking", with the split: 10 points lost to ranking, 34 to the threshold.

- **Related work added from a sweep on 2026-11-07, each checked on its arXiv or venue page:**
  - Yang et al. 2610.32102: few Euler steps keep the mean plan and shrink its spread. Cited qualitatively only; their ratios are radii and our score is squared.
  - Sujit et al. 2610.30735 finds 34.6% for released SmolVLA on LIBERO-10, and 1 step does not hurt success.
  - Doula & Mühlhäuser (NeurIPS 2026 E&D) is the LM analogue: conformal coverage under inference-time shift.
  - ActProbe 2606.08508 uses the same action-consistency signal.
  - FAIL-Detect (RSS 2025) calibrates on successes.
  - The concurrent paper is now public: Bysani, RoboPAD @ NeurIPS 2026.
  - QuantVLA is now CVPR 2026.
- **Baseline-gap footnote:** the released checkpoint's action expert is 0.5× the backbone width (the paper uses 0.75×). MuJoCo ≥3.4 (we used 3.8.1) changes the start scenes of 152/500 LIBERO-Long states (lerobot#4390).
- **Not in the paper, in case a reader asks:** the concurrent paper's abstract says int8 leaves the VLM text stream "numerically untouched", while we see a 0.29 SD int8 shift in the VLM probe. A likely difference is that we also quantize the vision encoder (all Linear layers except lm_head). Unverified.

## Original submission (2026-10-19)

- **Venue:** OpenReview, `robot-learning.org/CoRL/2026/Workshop/EEAI`.
- **Track:** new work.
- **Deadline:** 2026-10-19, 11:59 PM AoE.
- **Format:** 4-page extended abstract, plus references.
- **Review:** non-archival, double-blind. Concurrent submissions are allowed.
- **PDF:** `paper/eeai_submission.pdf`, anonymous.

## Title

Does a Robot's Failure Monitor Still Work After a Speed-Up?

## Abstract (camera-ready; paste into the OpenReview revision form)

Robots controlled by vision-language-action (VLA) models are often paired with a failure monitor: a lightweight check that raises an alarm when a task seems to be going wrong. The monitor's alarm threshold is tuned on the model as originally trained, but the deployed model is often a sped-up version, so we ask whether the monitor still works after that swap. We tune simplified versions of four common monitors on the open SmolVLA model, then run sped-up versions of it on the LIBERO-Long benchmark with the monitors left unchanged. Building each action plan in 2 refinement steps instead of 10 makes the model 2.5x faster with no observed drop in success (27% before, 32% after). Yet it disables a monitor that checks whether the robot's successive plans agree: that monitor caught about half of all failures before and catches none after, because the faster model's plans agree more closely, which pulls every score below the old threshold. Counting false alarms, the only thing the threshold's tuning guarantees, does not flag this, because false alarms drop to zero too. Rounding the model's weights to fewer bits nudges a different monitor, one that reads the model's visual and language features, toward reporting "safe". After any speed-up, a monitor should be re-tested; comparing its scores on successful runs before and after the speed-up already exposes the shift, without a single failure.

## Keywords

VLA, acceleration, flow matching, quantization, runtime monitoring, failure detection, conformal prediction

## TL;DR

Making SmolVLA 2.5x faster with fewer flow steps, with no observed drop in success, silently disables an action-consistency failure monitor: its false alarms drop too, so the usual check looks like an improvement.

## Camera-ready checklist

- [ ] Read `paper/main.pdf` once end to end.
- [x] Make the GitHub repo public (done 2026-11-07, camera-ready commit 7171a47, tag `eeai-camera-ready`). The paper links it. A secret scan of the tree and full history on 2026-11-07 was clean: Modal's HF token is referenced only by name, and the reference PDFs and `data/` were never committed. `NOTES.md` is public with it.
- [ ] Upload `paper/main.pdf` as the OpenReview revision and paste the abstract and TL;DR above.
- [ ] Poster for 2026-11-12 (and a 5-minute talk if selected). Sample videos and stills are in `media/` (see the README).
- [ ] Upload the videos where wanted (OpenReview supplementary, a project page, slides).
