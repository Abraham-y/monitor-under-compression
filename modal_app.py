"""GPU rollouts on Modal: SmolVLA on LIBERO under each compression condition.

    modal run modal_app.py --stage smoke --condition ref            # 2 episodes, prints a summary
    modal run --detach modal_app.py --stage collect --plan main     # every (condition, task) in PLANS
    modal run modal_app.py --stage fetch --run main                 # -> data/main/

Each (condition, task) runs in its own container; at most MAX_CONTAINERS at once (the
account caps at 10 GPUs). Re-running collect skips batches whose .npz already exists.
Everything persists on one Modal volume:

    /vol/hf                                   HF cache (checkpoint, SmolVLM2 backbone)
    /vol/runs/<run>/<condition>/<suite>/task<k>_init<a>-<b>_s<seed>.npz
"""

import json
import os
from pathlib import Path

import modal

APP_NAME = "monitor-under-compression"
VOLUME_NAME = "monitor-under-compression"
GPU = os.environ.get("MUC_GPU", "L4")
MAX_CONTAINERS = 10
ROOT = Path(__file__).resolve().parent

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
NVIDIA_EGL = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libegl1", "libgl1", "libgles2", "libglvnd0", "libglib2.0-0", "libosmesa6",
        "ffmpeg", "git", "build-essential", "cmake",
    )
    .pip_install("lerobot[smolvla,libero]==0.6.1")
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        f"echo '{NVIDIA_EGL}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "MUJOCO_GL": os.environ.get("MUC_GL", "egl"),
            "PYOPENGL_PLATFORM": os.environ.get("MUC_GL", "egl"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("rollout")
)

app = modal.App(APP_NAME, image=image)
KW = dict(
    gpu=GPU,
    cpu=8.0,
    memory=32 * 1024,
    volumes={"/vol": vol},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)

# (suite, task ids, init-state ids, batch size, seeds) per condition. See PREREG.md.
SUITE = "libero_10"
PLANS = {
    "main": {
        "ref": {"inits": list(range(50)), "seeds": [0]},
        "w8": {"inits": list(range(30, 50)), "seeds": [0]},
        "w4": {"inits": list(range(30, 50)), "seeds": [0]},
        "steps2": {"inits": list(range(30, 50)), "seeds": [0]},
    },
}
BATCH = 10


def _batches(inits: list[int], b: int) -> list[list[int]]:
    return [inits[i : i + b] for i in range(0, len(inits), b)]


def _npz_name(task_id: int, init_ids: list[int], seed: int) -> str:
    return f"task{task_id}_init{init_ids[0]}-{init_ids[-1]}_s{seed}.npz"


def _summary(out: dict) -> dict:
    return {
        "success": out["success"].tolist(),
        "length": out["length"].tolist(),
        "wall_s": round(float(out["wall_s"]), 1),
        "s_per_call": round(float(out["wall_s"]) / int(out["n_policy_calls"]), 3),
        "shapes": {k: list(out[k].shape) for k in ("h_vlm", "h_exp", "chunk", "action")},
    }


@app.function(**KW)
def smoke(condition: str, suite: str, task_id: int, n_envs: int) -> dict:
    import subprocess

    import rollout

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.strip()
    r = rollout.Runner(suite, task_id, n_envs, condition)
    out = r.run_batch(list(range(n_envs)), seed=0)
    path = Path(f"/vol/runs/smoke/{condition}/{suite}/{_npz_name(task_id, list(range(n_envs)), 0)}")
    meta = {"condition": condition, "suite": suite, "task_id": task_id, "task": r.task_description}
    rollout.save_npz(path, out, meta)
    vol.commit()
    r.close()
    return {
        "gpu": gpu,
        "task": r.task_description,
        "num_steps": r.num_steps,
        "n_action_steps": r.policy.config.n_action_steps,
        "compression": r.compression,
        "verify": r.verify,
        **_summary(out),
    }


@app.function(**KW, max_containers=MAX_CONTAINERS)
def collect(run: str, condition: str, suite: str, task_id: int, inits: list[int], seeds: list[int]) -> dict:
    import rollout

    vol.reload()
    todo = []
    for seed in seeds:
        for ids in _batches(inits, BATCH):
            path = Path(f"/vol/runs/{run}/{condition}/{suite}/{_npz_name(task_id, ids, seed)}")
            if not path.exists():
                todo.append((ids, seed, path))
    if not todo:
        return {"condition": condition, "task_id": task_id, "skipped": True}
    r = rollout.Runner(suite, task_id, BATCH, condition)
    meta = {"condition": condition, "suite": suite, "task_id": task_id, "task": r.task_description,
            "compression": r.compression, "verify": r.verify, "num_steps": r.num_steps}
    done = []
    for ids, seed, path in todo:
        out = r.run_batch(ids, seed)
        rollout.save_npz(path, out, meta)
        vol.commit()
        done.append({"init": ids, "seed": seed, **_summary(out)})
        print(json.dumps({"condition": condition, "task_id": task_id, **done[-1]}), flush=True)
    r.close()
    return {"condition": condition, "task_id": task_id, "batches": done}


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    condition: str = "ref",
    suite: str = SUITE,
    task_id: int = 0,
    n_envs: int = 2,
    plan: str = "main",
    conditions: str = "",
    tasks: str = "",
    run: str = "",
):
    if stage == "smoke":
        print(json.dumps(smoke.remote(condition, suite, task_id, n_envs), indent=2))
    elif stage == "collect":
        spec = PLANS[plan]
        conds = conditions.split(",") if conditions else list(spec)
        task_ids = [int(t) for t in tasks.split(",")] if tasks else list(range(10))
        args = [(run or plan, c, suite, t, spec[c]["inits"], spec[c]["seeds"]) for c in conds for t in task_ids]
        for res in collect.starmap(args, return_exceptions=True):
            print(json.dumps(res if isinstance(res, dict) else {"error": repr(res)})[:2000])
    elif stage == "fetch":
        import subprocess

        dest = ROOT / "data" / (run or plan)
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME, f"runs/{run or plan}", str(dest.parent)],
                       check=True)
    else:
        raise ValueError(stage)
