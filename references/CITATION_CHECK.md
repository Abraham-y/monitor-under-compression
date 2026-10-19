# Citation check (2026-10-19)

Every work cited in `paper/main.tex` was downloaded to `references/pdfs/` (arXiv) or
`references/web/`. Each claim the paper attributes to it was checked against the text. The
PDFs are not committed; see `.gitignore`. Re-download them with the arXiv IDs below.

The concurrent paper is the one exception. OpenReview gates it behind a browser check that
we cannot complete. Its claims were checked against the full text the user pasted into the
working session on 2026-10-18; no local PDF exists yet. To get the PDF, open
openreview.net/forum?id=atOkXTzFCL in a browser.

| key | source | claim in paper | verified against | result |
|---|---|---|---|---|
| smolvla | arXiv 2506.01844 | ~450M params; SmolVLM-2 backbone; flow-matching action expert; 71% on LIBERO-Long | Table 2 ("SmolVLA (0.45B) … 90 96 92 71"); §3 (SmolVLM-2 backbone, action expert trained with flow matching) | ✓ |
| libero | arXiv 2306.03310 | LIBERO-Long is a suite of 10 long-horizon tasks | §4: the long-horizon suite is named LIBERO-LONG; suites have 10 tasks | ✓. "50 fixed initial states per task" is from the benchmark code, not the paper; now worded as our setup |
| lerobot#3264 | github.com/huggingface/lerobot/issues/3264 (JSON saved) | a public reproduction of this checkpoint reports 56% on LIBERO-Long | issue body: `libero_10 56.0%`, lerobot 0.5.1 | ✓. It is one reproduction, so the paper says "a public reproduction" |
| vlaquantbench | arXiv 2609.25376 | VLAs are accelerated by quantization; acceleration studies measure success and speed | abstract: closed-loop PTQ evaluation; real-kernel measurements | ✓ |
| quantvla | arXiv 2602.20309 | quantization for VLAs; measures success, compute, memory | abstract and intro | ✓ |
| care | arXiv 2610.08917 | acceleration includes pruning and fewer denoising steps; acceleration can break tasks that average success hides | intro: visual-token pruning, action chunking, reduced flow-integration steps for π0.5; abstract on failures hidden by average metrics | ✓ |
| embodiedeff | arXiv 2603.19131 | compression changes motion quality at similar success | abstract: completion time, smoothness, joint rotation, energy | ✓ |
| safe | arXiv 2506.09937 | VLA monitor with threshold calibrated on rollouts; learned probe on internal features; per-step labels from episode outcome; time-varying conformal bands calibrated on successful rollouts | §3–4: MLP/LSTM on VLA features, loss on all timesteps with rollout failure label, functional CP band from successful rollouts in a calibration set (one-sided) | ✓. Ours is a logistic probe, so the paper says "a linear version of SAFE's learned probe" |
| safecast | arXiv 2608.04246 | studies threshold/calibration drift under deployment-time input shift | abstract: clutter, distractors, lighting, novel objects, instructions; contrast-set calibration | ✓ |
| vlafail | arXiv 2606.21386 | last-layer Mahalanobis without failure data; consecutive action-chunk consistency; calibrated threshold | abstract and §1 | ✓ |
| sentinel | arXiv 2410.04640 | monitors temporal consistency of action chunks; calibrated on successful rollouts | abstract (STAC); appendix (calibration on M successful rollouts); STAC uses MMD | ✓. Ours is a single-sample squared difference, hence "Sentinel-style" |
| fiper | arXiv 2510.09459 | action-chunk statistics; calibrated with conformal prediction on successful rollouts | abstract: action-chunk entropy; scores calibrated on successful rollouts via CP | ✓ |
| rescue | arXiv 2610.13516 | evaluates monitors by the interventions their alarms trigger | Fig. 1 and abstract: counterfactual intervention evaluation | ✓ |
| conformal | arXiv 2107.07511 | conformal guarantees require exchangeability of calibration and test points | §on exchangeability | ✓ |
| duan | arXiv 2606.15980 | LM probes go stale under fine-tuning/LoRA but quantization leaves AUROC largely intact; staleness is recoverable drift, not erased signal | abstract and intro: quantization has low big-drop rates and no operational failures; LoRA 43–54% big-drop; realignment repairs stale cells, "recoverable representational drift rather than signal erasure" | ✓. **Fixed:** the old intro sentence ("updates can preserve ranking while moving scores") overstated Duan; reworded |
| rankpreserving | OpenReview atOkXTzFCL (anonymous NeurIPS 2026 workshop submission) | SmolVLA probe under simulated int4; open-loop replay of fp32 observations; coverage shifts at unchanged AUC; scoring at episode end leaks episode length | full text: §3 (paired activations by replaying recorded actions), Table 2–3, §3 "Evaluation horizon" | ✓ (against the user-pasted full text; no local PDF) |

Uncited items still in `refs.bib` (`saequant`, `onnxsmolvla`) do not appear in the PDF, since
BibTeX drops uncited entries.
