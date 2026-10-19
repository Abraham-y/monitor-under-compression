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

Robots controlled by vision-language-action (VLA) models are often paired with a failure monitor: a lightweight check that raises an alarm when a task seems to be going wrong. The monitor's alarm threshold is tuned on the model as originally trained, but the model that gets deployed is usually a sped-up version, so we ask whether the monitor still works after that swap. We tune four common monitors on the open SmolVLA model, then run sped-up versions of it in the LIBERO-Long simulator with the monitors left unchanged. The most practical speed-up, building each action plan in 2 refinement steps instead of 10, makes the model 2.5x faster and just as successful. Yet it disables a monitor that watches the robot's planned actions: it caught about half of all failures before and catches none after, because the faster model's successive plans agree more closely, which pulls its scores below the old threshold. Checking false alarms, the only thing the monitor's tuning guarantees, cannot reveal this, since false alarms drop to zero too. Compressing the model's weights has a mirror-image effect on a monitor that reads the model's visual and language features, pushing it toward reporting "safe" as compression increases. Whenever a robot policy is sped up, its monitor should be re-tuned and tested on real failures.

## Keywords

VLA, acceleration, flow matching, quantization, runtime monitoring, failure detection, conformal prediction

## TL;DR

Two-step flow sampling makes SmolVLA 2.5x faster at equal success, but silently disarms its
action-side failure monitors. A false-alarm check passes while detection drops to zero.

## Checklist before submitting

- [ ] Read the PDF once end to end.
- [ ] An OpenReview profile exists for every author, so the form will accept the author list.
- [ ] Decide whether the anonymous repo link goes in supplementary material. Today it is a private GitHub repo, and not anonymous.
