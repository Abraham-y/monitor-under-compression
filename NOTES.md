# NOTES

## 2026-10-17

- Repo created. lerobot 0.6.1, checkpoint `HuggingFaceVLA/smolvla_libero`
  (chunk 50, n_action_steps 1, 10 Euler steps, SmolVLM2-500M backbone, expert width 0.5).
- Novelty sweep (web + arXiv keyword, 2026-10-13): no paper evaluates a VLA failure
  monitor on a compressed version of its policy. Closest: SAFECAST (calibration under
  input shift), VLAQuantBench (PTQ success only), arXiv 2606.03002 (SAE features
  degrade under LLM quantization while perplexity holds).
- Synthetic tests for monitors.py pass (tests/test_monitors.py).
