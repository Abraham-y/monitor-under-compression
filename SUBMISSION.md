# EEAI @ CoRL 2026 submission

- **Venue:** OpenReview, `robot-learning.org/CoRL/2026/Workshop/EEAI`.
- **Track:** new work.
- **Deadline:** 2026-10-19, 11:59 PM AoE.
- **Format:** 4-page extended abstract, plus references.
- **Review:** non-archival, double-blind. Concurrent submissions are allowed.
- **PDF:** `paper/main.pdf`. It is anonymous: no repo link and no names.

## Title

Does a Robot's Failure Monitor Still Work After a Speed-Up?

## Abstract (plain text for the form)

Robots controlled by vision-language-action (VLA) models are often paired with a failure monitor: a lightweight check that raises an alarm when a task seems to be going wrong. The monitor's alarm threshold is tuned on the model as originally trained, but the deployed model is often a sped-up version, so we ask whether the monitor still works after that swap. We tune four common monitors on the open SmolVLA model, then run sped-up versions of it in the LIBERO-Long simulator with the monitors left unchanged. Building each action plan in 2 refinement steps instead of 10 makes the model 2.5x faster with no loss of success in our test (32% vs. 27%). Yet it disables a monitor that checks whether the robot's successive plans agree: that monitor caught about half of all failures before and catches none after, because the faster model's plans agree more closely, which pulls every score below the old threshold. Counting false alarms, the only thing the threshold's tuning guarantees, does not flag this, because false alarms drop to zero too. Rounding the model's weights to fewer bits nudges a different monitor, one that reads the model's visual and language features, toward reporting "safe". After any speed-up, a monitor should be re-tested; comparing its scores on successful runs before and after the speed-up already exposes the shift, without a single failure.

## Keywords

VLA, acceleration, flow matching, quantization, runtime monitoring, failure detection, conformal prediction

## TL;DR

Making SmolVLA 2.5x faster with fewer flow steps keeps its success rate but silently disables an action-consistency failure monitor: its false alarms drop too, so the usual check looks like an improvement.

## Checklist before submitting

- [ ] Read the PDF once end to end.
- [ ] An OpenReview profile exists for every author, so the form will accept the author list.
- [ ] Decide whether the anonymous repo link goes in supplementary material. Today it is a private GitHub repo, and not anonymous.
