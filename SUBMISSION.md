# EEAI @ CoRL 2026 submission

- **Venue:** OpenReview, `robot-learning.org/CoRL/2026/Workshop/EEAI`.
- **Track:** new work.
- **Deadline:** 2026-10-19, 11:59 PM AoE.
- **Format:** 4-page extended abstract, plus references.
- **Review:** non-archival, double-blind. Concurrent submissions are allowed.
- **PDF:** `paper/main.pdf`. It is anonymous: no repo link and no names.

## Title

Recalibrate After You Accelerate: Few-Step Flow Sampling Silently Disarms a VLA's Failure Monitors

## Abstract (plain text for the form)

Failure monitors for vision-language-action (VLA) policies are calibrated on the policy as trained, but the deployed policy is usually accelerated. We freeze four monitors on SmolVLA, calibrate their alarms on reference rollouts, and run compressed policies closed loop on LIBERO-Long. Sampling the flow-matching action expert with 2 Euler steps instead of 10 cuts policy latency 2.5x at unchanged task success. It also shrinks action-chunk disagreement by a uniform 0.42x, so an action-consistency monitor that caught 49% of failures catches none of 68, and an expert-state probe loses a quarter of its detections. Monitors that read the vision-language backbone, which step reduction leaves untouched, are unaffected. Because conformal thresholds calibrated on successes bound only false alarms, the broken monitors pass a false-alarm check. Weight quantization shows the mirror image: it moves a backbone probe toward "looks safe" on identical inputs (-0.3, -2.3 and -4.6 SD at int8 and two int4 settings), while its detection falls from 0.67 to 0.54, 0.09 and 0.

## Keywords

VLA, acceleration, flow matching, quantization, runtime monitoring, failure detection, conformal prediction

## TL;DR

Two-step flow sampling makes SmolVLA 2.5x faster at equal success, but silently disarms its
action-side failure monitors. A false-alarm check passes while detection drops to zero.

## Checklist before submitting

- [ ] Read the PDF once end to end.
- [ ] An OpenReview profile exists for every author, so the form will accept the author list.
- [ ] Decide whether the anonymous repo link goes in supplementary material. Today it is a private GitHub repo, and not anonymous.
