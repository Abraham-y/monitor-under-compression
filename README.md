# Does a VLA failure monitor survive compression?

Runtime failure monitors for vision-language-action policies are calibrated on the
full-precision policy. Deployed policies are compressed. This repo checks whether the
monitor's guarantees and accuracy carry over: SmolVLA on LIBERO-Long, four monitors
(two hidden-state probes, a Mahalanobis OOD score, action-chunk consistency), four
policy conditions (reference, int8 and int4 weight-only RTN, 2-step flow sampling).

Target: CoRL 2026 workshop on Efficient Foundation Models for Real-Time Embodied AI
(EEAI), 4-page extended abstract, due 2026-10-19 AoE.

- `PREREG.md` — hypotheses, splits, monitors, decision rules (frozen before data)
- `rollout.py` — LeRobot rollouts with compression and feature hooks (runs on Modal)
- `modal_app.py` — smoke / collect / fetch stages
- `monitors.py`, `analyze.py` — monitors, conformal thresholds, evaluation
- `NOTES.md` — running log
