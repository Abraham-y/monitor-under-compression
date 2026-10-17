"""Failure monitors fit on the reference policy, then scored on every condition.

Analysis plan: PREREG.md. Split by LIBERO init state (identical for every task):

    train   init 0-29    reference only. Fits the probes and the Mahalanobis model.
    calib   init 30-39   reference sets the deployed threshold; each compressed condition's
                         own calib episodes give the "recalibrated" comparison.
    test    init 40-49   every condition. All reported numbers come from here.

Monitors, each producing a per-step score s_t; the episode score at t is the running mean
S_t = mean(s_1..s_t), as in SAFE (Gu et al. 2025):

    probe_vlm   logistic probe on the VLM prefix state (h_vlm), failure = 1
    probe_exp   logistic probe on the action-expert state (h_exp)
    maha_vlm    Mahalanobis distance of h_vlm to successful training steps (no failure labels)
    act_cons    disagreement between consecutive action chunks on their 49 shared steps
                (no training; the action-only signal of Sentinel / VLA-FAIL)

Alarm: first t >= BURN_IN with S_t > tau. tau is the conformal (1 - ALPHA) quantile of
max_{t >= BURN_IN} S_t over successful calibration episodes, so under exchangeability a new
successful episode raises a false alarm with probability <= ALPHA.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHA = 0.10
BURN_IN = 10
STEP_STRIDE = 4  # subsample steps when fitting (consecutive steps are near-duplicates)
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
MAHA_PCS = 64
TRAIN, CALIB, TEST = range(0, 30), range(30, 40), range(40, 50)
MONITORS = ("probe_vlm", "probe_exp", "maha_vlm", "act_cons")


@dataclass
class Episode:
    condition: str
    task_id: int
    init: int
    seed: int
    success: bool
    h_vlm: np.ndarray  # (T, 960) float16
    h_exp: np.ndarray  # (T, 480)
    chunk: np.ndarray  # (T, 50, 7)

    @property
    def length(self) -> int:
        return self.h_vlm.shape[0]


def load_run(root: Path) -> list[Episode]:
    """Every episode under data/<run>/<condition>/<suite>/*.npz, truncated to its length."""
    eps = []
    for f in sorted(root.glob("*/*/*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = ast.literal_eval(str(z["meta"]))
        for b, init in enumerate(z["init_ids"]):
            L = int(z["length"][b])
            eps.append(
                Episode(
                    condition=meta["condition"],
                    task_id=int(meta["task_id"]),
                    init=int(init),
                    seed=int(z["seed"]),
                    success=bool(z["success"][b]),
                    h_vlm=z["h_vlm"][b, :L],
                    h_exp=z["h_exp"][b, :L],
                    chunk=z["chunk"][b, :L],
                )
            )
    return eps


def select(eps, condition, inits):
    return [e for e in eps if e.condition == condition and e.init in inits]


# ---------------------------------------------------------------- per-step scores


def chunk_disagreement(chunk: np.ndarray) -> np.ndarray:
    """s_t = mean sq. difference of chunk_t[k] and chunk_{t-1}[k+1]; s_0 copies s_1."""
    c = chunk.astype(np.float32)
    d = ((c[1:, :-1] - c[:-1, 1:]) ** 2).mean(axis=(1, 2))
    return np.concatenate([d[:1], d]) if len(d) else np.zeros(1, np.float32)


class Probe:
    def __init__(self, key: str):
        self.key = key

    def fit(self, eps: list[Episode], seed: int = 0):
        X = np.concatenate([getattr(e, self.key)[::STEP_STRIDE] for e in eps]).astype(np.float32)
        y = np.concatenate([np.full(len(getattr(e, self.key)[::STEP_STRIDE]), not e.success) for e in eps])
        g = np.concatenate([np.full(len(getattr(e, self.key)[::STEP_STRIDE]), i) for i, e in enumerate(eps)])
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        # C by grouped CV on episode-level AUROC of the running-mean score
        best, self.cv = None, {}
        for C in C_GRID:
            aucs = []
            for tr, va in GroupKFold(n_splits=5).split(Xs, y, g):
                clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xs[tr], y[tr])
                va_eps = [eps[i] for i in np.unique(g[va])]
                lab = [not e.success for e in va_eps]
                if len(set(lab)) < 2:
                    continue
                sc = [episode_score(self._raw(clf, e)) for e in va_eps]
                aucs.append(roc_auc_score(lab, sc))
            self.cv[C] = float(np.mean(aucs)) if aucs else float("nan")
            if best is None or self.cv[C] > self.cv[best]:
                best = C
        self.C = best
        self.clf = LogisticRegression(C=best, max_iter=3000, class_weight="balanced").fit(Xs, y)
        return self

    def _raw(self, clf, e: Episode) -> np.ndarray:
        return clf.decision_function(self.scaler.transform(getattr(e, self.key).astype(np.float32)))

    def steps(self, e: Episode) -> np.ndarray:
        return self._raw(self.clf, e)


class Mahalanobis:
    def __init__(self, key: str):
        self.key = key

    def fit(self, eps: list[Episode], seed: int = 0):
        X = np.concatenate([getattr(e, self.key)[::STEP_STRIDE] for e in eps if e.success]).astype(np.float32)
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=MAHA_PCS, random_state=seed).fit(self.scaler.transform(X))
        Z = self.pca.transform(self.scaler.transform(X))
        self.lw = LedoitWolf().fit(Z)
        return self

    def steps(self, e: Episode) -> np.ndarray:
        Z = self.pca.transform(self.scaler.transform(getattr(e, self.key).astype(np.float32)))
        return self.lw.mahalanobis(Z)


class ActionConsistency:
    def fit(self, eps, seed: int = 0):
        return self

    def steps(self, e: Episode) -> np.ndarray:
        return chunk_disagreement(e.chunk)


def build_monitors(train_eps: list[Episode], seed: int = 0) -> dict:
    return {
        "probe_vlm": Probe("h_vlm").fit(train_eps, seed),
        "probe_exp": Probe("h_exp").fit(train_eps, seed),
        "maha_vlm": Mahalanobis("h_vlm").fit(train_eps, seed),
        "act_cons": ActionConsistency().fit(train_eps, seed),
    }


# ---------------------------------------------------------------- episode-level


def running_mean(s: np.ndarray) -> np.ndarray:
    return np.cumsum(s) / np.arange(1, len(s) + 1)


def episode_score(s: np.ndarray) -> float:
    """max_{t >= BURN_IN} S_t; episodes shorter than the burn-in use their last S_t."""
    S = running_mean(s)
    return float(S[BURN_IN:].max()) if len(S) > BURN_IN else float(S[-1])


def conformal_tau(calib_success_scores: list[float], alpha: float = ALPHA) -> float:
    n = len(calib_success_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(calib_success_scores)[k - 1])


def first_alarm(s: np.ndarray, tau: float) -> int | None:
    S = running_mean(s)
    idx = np.nonzero(S[BURN_IN:] > tau)[0]
    return int(idx[0] + BURN_IN) if len(idx) else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def evaluate(monitor, test_eps: list[Episode], tau: float) -> dict:
    succ = [e for e in test_eps if e.success]
    fail = [e for e in test_eps if not e.success]
    s_steps = {id(e): monitor.steps(e) for e in test_eps}
    fa = [first_alarm(s_steps[id(e)], tau) is not None for e in succ]
    det = [first_alarm(s_steps[id(e)], tau) for e in fail]
    out = {
        "n_success": len(succ),
        "n_fail": len(fail),
        "far": float(np.mean(fa)) if succ else float("nan"),
        "far_ci": wilson(int(np.sum(fa)), len(succ)),
        "tpr": float(np.mean([d is not None for d in det])) if fail else float("nan"),
        "tpr_ci": wilson(int(sum(d is not None for d in det)), len(fail)),
        "median_alarm_frac": float(np.median([d / e.length for d, e in zip(det, fail) if d is not None]))
        if any(d is not None for d in det)
        else float("nan"),
    }
    if succ and fail:
        lab = [0] * len(succ) + [1] * len(fail)
        sc = [episode_score(s_steps[id(e)]) for e in succ + fail]
        out["auroc"] = float(roc_auc_score(lab, sc))
    else:
        out["auroc"] = float("nan")
    return out
