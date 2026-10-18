"""End to end on synthetic files in rollout.py's on-disk format: load_run -> analyze."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, cond: str, task: int, inits: list[int], rng, shift: float):
    B, T = len(inits), 120
    succ = rng.random(B) > 0.4
    length = np.where(succ, rng.integers(60, 100, B), T).astype(np.int32)
    mu = np.where(succ, 0.0, 0.7)[:, None, None] + shift
    out = {
        "h_vlm": (rng.normal(size=(B, T, 24)) + mu).astype(np.float16),
        "h_exp": (rng.normal(size=(B, T, 12)) + mu).astype(np.float16),
        "chunk": rng.normal(0, 0.1, size=(B, T, 50, 7)).astype(np.float16),
        "action": np.zeros((B, T, 7), np.float32),
        "length": length, "success": succ, "init_ids": np.array(inits, np.int32),
        "seed": np.int64(0), "wall_s": np.float64(1), "policy_s": np.float64(1), "n_policy_calls": np.int64(T),
    }
    meta = {"condition": cond, "suite": "libero_10", "task_id": task}
    p = root / cond / "libero_10" / f"task{task}_init{inits[0]}-{inits[-1]}_s0.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out, meta=np.array(repr(meta)))


def test_analyze_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    run = "synthetic_test"
    root = ROOT / "data" / run
    for cond, inits, shift in (("ref", range(50), 0.0), ("w8", range(30, 50), 0.05),
                               ("w4", range(30, 50), 0.6), ("steps2", range(30, 50), 0.0)):
        for task in range(3):
            for i in range(0, len(inits), 10):
                _write(root, cond, task, list(inits)[i:i + 10], rng, shift)
    env = {"MUC_MAHA_PCS": "8"}
    r = subprocess.run([sys.executable, "analyze.py", "--run", run], cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, **env})
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((ROOT / "results" / run / "results.json").read_text())
    m = res["monitors"]["maha_vlm"]
    assert m["w4"]["far"] > m["ref"]["far"]
    import shutil
    shutil.rmtree(root)
    shutil.rmtree(ROOT / "results" / run)
