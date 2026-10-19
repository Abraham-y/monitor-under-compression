"""LIBERO rollouts of SmolVLA under a compression condition, logging what monitors read.

Runs inside the Modal image (modal_app.py). Reuses LeRobot's own config parsing, env,
processors and rollout loop (lerobot 0.6.1, scripts/lerobot_eval.py) so task success
matches `lerobot-eval`; the only additions are compression and forward hooks.

Per policy call (one per env step: the checkpoint replans every step, n_action_steps=1)
the hooks record, for every env in the batch:

    h_vlm   VLM prefix output after the final norm, mean-pooled over valid tokens (960)
    h_exp   action-expert output at the last denoising step, mean-pooled over the chunk (480)
    chunk   the normalized action chunk, first 7 dims (50 x 7)

Conditions (see PREREG.md):
    ref      the checkpoint as shipped
    w8       RTN weight-only int8, per output channel, symmetric
    w4       RTN weight-only int4, groups of 128 along the input dim, asymmetric
    steps2   flow-matching Euler steps 10 -> 2 (weights untouched)
    w4nv     w4 with the vision encoder left in full precision (post hoc, exploratory;
             added after w4 collapsed task success, see NOTES.md)

Weight quantization is simulated (quantize then dequantize in place), the usual PTQ
accuracy setup. It changes the numerics a real int8/int4 kernel would see, but not
latency; `verify_quantization` checks the weights really sit on the int grid.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

POLICY = "HuggingFaceVLA/smolvla_libero"
CONDITIONS = ("ref", "w8", "w4", "steps2", "w4nv")
QUANT = ("w8", "w4", "w4nv")
# Linear layers inside these modules are quantized; the policy's own small heads
# (state_proj, action_in/out_proj, action_time_mlp_*) stay in full precision.
QUANT_ROOT = "model.vlm_with_expert"
W4_GROUP = 128


def build_cfg(suite: str, task_id: int, n_envs: int, condition: str, episode_length: int | None = None):
    """An EvalPipelineConfig built exactly as `lerobot-eval` would build it from the CLI."""
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig

    argv = [
        "lerobot-eval",
        f"--policy.path={POLICY}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids=[{task_id}]",
        f"--eval.batch_size={n_envs}",
        f"--eval.n_episodes={n_envs}",
        "--eval.use_async_envs=false",
        "--env.max_parallel_tasks=1",
    ]
    if condition == "steps2":
        argv.append("--policy.num_steps=2")
    if episode_length:
        argv.append(f"--env.episode_length={episode_length}")

    def _get(cfg):
        return cfg

    # parser.wrap reads the real class from the annotation; this module's postponed
    # annotations would hand it a string.
    _get.__annotations__ = {"cfg": EvalPipelineConfig}
    _get = parser.wrap()(_get)

    old = sys.argv
    sys.argv = argv
    try:
        return _get()
    finally:
        sys.argv = old


# ---------------------------------------------------------------- compression


def _quant_linears(policy: nn.Module, condition: str = "") -> list[tuple[str, nn.Linear]]:
    root = policy.get_submodule(QUANT_ROOT)
    return [
        (f"{QUANT_ROOT}.{n}", m)
        for n, m in root.named_modules()
        if isinstance(m, nn.Linear) and not n.endswith("lm_head")
        and not (condition == "w4nv" and "vision_model" in n)
    ]


@torch.no_grad()
def _rtn_int8_per_channel(w: torch.Tensor) -> torch.Tensor:
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.clamp(torch.round(wf / scale), -127, 127)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def _rtn_int4_group(w: torch.Tensor, group: int = W4_GROUP) -> torch.Tensor:
    out_f, in_f = w.shape
    g = group if in_f % group == 0 else in_f  # fall back to one group per row
    wf = w.float().reshape(out_f, in_f // g, g)
    lo = wf.amin(dim=-1, keepdim=True)
    hi = wf.amax(dim=-1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-8) / 15.0
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(wf / scale) + zero, 0, 15)
    return ((q - zero) * scale).reshape(out_f, in_f).to(w.dtype)


def apply_compression(policy: nn.Module, condition: str) -> dict:
    """Quantize in place for w8/w4. steps2 is set through the config in build_cfg."""
    info = {"condition": condition, "n_linear": 0, "n_params": 0}
    if condition not in QUANT:
        return info
    fn = _rtn_int8_per_channel if condition == "w8" else _rtn_int4_group
    for _, m in _quant_linears(policy, condition):
        m.weight.data.copy_(fn(m.weight.data))
        info["n_linear"] += 1
        info["n_params"] += m.weight.numel()
    return info


@torch.no_grad()
def verify_quantization(policy: nn.Module, condition: str, sample: int = 12) -> dict:
    """Check that quantized weights take at most 2^bits distinct values per group.

    Guards against the failure in arXiv 2609.14146, where 'INT8' artifacts were FP32.
    """
    lins = _quant_linears(policy, condition)
    total = sum(m.weight.numel() for _, m in policy.named_modules() if isinstance(m, nn.Linear))
    covered = sum(m.weight.numel() for _, m in lins)
    out = {"linear_params_total": int(total), "linear_params_targeted": int(covered)}
    if condition not in QUANT:
        return out
    worst = 0
    idx = np.linspace(0, len(lins) - 1, num=min(sample, len(lins))).astype(int)
    for i in idx:
        w = lins[i][1].weight.float()
        if condition == "w8":
            rows = w[: min(8, w.shape[0])]
            worst = max(worst, max(len(torch.unique(r)) for r in rows))
        else:
            in_f = w.shape[1]
            g = W4_GROUP if in_f % W4_GROUP == 0 else in_f
            groups = w[:4].reshape(-1, g)
            worst = max(worst, max(len(torch.unique(r)) for r in groups))
    out["max_unique_per_group"] = int(worst)
    out["limit"] = 255 if condition == "w8" else 16
    out["ok"] = bool(worst <= out["limit"])
    return out


# ---------------------------------------------------------------- hooks


@dataclass
class Recorder:
    """Collects per-call features from the SmolVLA flow-matching model."""

    h_vlm: list
    h_exp: list
    chunk: list
    policy_s: float = 0.0

    @classmethod
    def attach(cls, policy: nn.Module) -> "Recorder":
        rec = cls([], [], [])
        model = policy.model
        vwe = model.vlm_with_expert
        state = {"pad": None, "exp": None}

        orig_embed_prefix = model.embed_prefix

        def embed_prefix(*a, **k):
            embs, pad_masks, att_masks = orig_embed_prefix(*a, **k)
            state["pad"] = pad_masks
            return embs, pad_masks, att_masks

        orig_forward = vwe.forward

        def forward(*a, **k):
            outs, pkv = orig_forward(*a, **k)
            prefix_out, suffix_out = outs[0], outs[1]
            if prefix_out is not None:
                m = state["pad"].to(prefix_out.dtype).unsqueeze(-1)
                pooled = (prefix_out * m).sum(1) / m.sum(1).clamp(min=1)
                rec.h_vlm.append(pooled.float().cpu().numpy().astype(np.float16))
            if suffix_out is not None:
                state["exp"] = suffix_out[:, -model.config.chunk_size :].float().mean(1)
            return outs, pkv

        orig_sample = model.sample_actions

        def sample_actions(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            actions = orig_sample(*a, **k)
            torch.cuda.synchronize()
            rec.policy_s += time.perf_counter() - t0
            rec.h_exp.append(state["exp"].cpu().numpy().astype(np.float16))
            rec.chunk.append(actions[:, :, :7].float().cpu().numpy().astype(np.float16))
            return actions

        model.embed_prefix = embed_prefix
        vwe.forward = forward
        model.sample_actions = sample_actions
        return rec

    def clear(self):
        self.policy_s = 0.0
        self.h_vlm.clear()
        self.h_exp.clear()
        self.chunk.clear()

    def stacked(self) -> dict[str, np.ndarray]:
        # lists of (B, ...) per step -> (B, T, ...)
        return {
            "h_vlm": np.stack(self.h_vlm, axis=1),
            "h_exp": np.stack(self.h_exp, axis=1),
            "chunk": np.stack(self.chunk, axis=1),
        }


# ---------------------------------------------------------------- driver


class Runner:
    """Loads the policy once for (suite, task, condition); runs batches of init states."""

    def __init__(self, suite: str, task_id: int, n_envs: int, condition: str,
                 episode_length: int | None = None, use_async: bool = False):
        assert condition in CONDITIONS, condition
        from lerobot.envs import make_env, make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors

        self.suite, self.task_id, self.n_envs, self.condition = suite, task_id, n_envs, condition
        cfg = build_cfg(suite, task_id, n_envs, condition, episode_length)
        self.cfg = cfg
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        envs = make_env(cfg.env, n_envs=n_envs, use_async_envs=use_async)
        self.vec_env = envs[suite][task_id]
        self.task_description = list(self.vec_env.call("task_description"))[0]

        self.policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        self.policy.eval()
        self.compression = apply_compression(self.policy, condition)
        self.verify = verify_quantization(self.policy, condition)
        if condition in QUANT and not self.verify["ok"]:
            raise RuntimeError(f"quantization check failed: {self.verify}")
        self.num_steps = self.policy.config.num_steps

        overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=overrides,
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        self.rec = Recorder.attach(self.policy)

    def run_batch(self, init_ids: list[int], seed: int) -> dict:
        from lerobot.scripts.lerobot_eval import rollout

        assert len(init_ids) == self.n_envs
        self.vec_env.set_attr("init_state_id", list(init_ids))
        torch.manual_seed(seed)
        np.random.seed(seed)
        seeds = [seed * 1000 + i for i in init_ids]
        self.rec.clear()
        t0 = time.time()
        with torch.no_grad():
            ret = rollout(
                env=self.vec_env,
                policy=self.policy,
                env_preprocessor=self.env_pre,
                env_postprocessor=self.env_post,
                preprocessor=self.pre,
                postprocessor=self.post,
                seeds=seeds,
            )
        wall = time.time() - t0
        feats = self.rec.stacked()
        done = ret["done"].numpy()  # (B, T) cumulative
        succ = ret["success"].numpy()  # (B, T)
        T = done.shape[1]
        # the first True in `done` is the last real step of that episode
        length = np.where(done.any(1), done.argmax(1) + 1, T)
        success = np.array([succ[b, : length[b]].any() for b in range(len(init_ids))])
        assert feats["h_vlm"].shape[1] == T, (feats["h_vlm"].shape, T)
        return {
            **feats,
            "action": ret["action"].numpy().astype(np.float32),
            "length": length.astype(np.int32),
            "success": success,
            "init_ids": np.array(init_ids, dtype=np.int32),
            "seed": np.int64(seed),
            "wall_s": np.float64(wall),
            "policy_s": np.float64(self.rec.policy_s),
            "n_policy_calls": np.int64(T),
        }

    def close(self):
        self.vec_env.close()


def save_npz(path: Path, out: dict, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out, meta=np.array(repr(meta)))
