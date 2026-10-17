"""Synthetic checks: probes separate a planted failure signal; conformal tau holds FAR."""

import numpy as np

import monitors as M


def _ep(cond, init, success, rng, shift=0.0, T=None):
    T = T or (120 if success else 300)
    mu = (0.0 if success else 0.8) + shift
    h = rng.normal(mu, 1.0, size=(T, 32)).astype(np.float16)
    he = rng.normal(mu, 1.0, size=(T, 16)).astype(np.float16)
    ch = rng.normal(0, 0.1 if success else 0.3, size=(T, 50, 7)).astype(np.float16)
    return M.Episode(cond, init % 10, init, 0, success, h, he, ch)


def _set(cond, inits, rng, shift=0.0):
    return [_ep(cond, i, (i * 7 + t) % 10 >= 3, rng, shift) for i in inits for t in range(3)]


def test_monitors_and_conformal():
    rng = np.random.default_rng(0)
    M.MAHA_PCS = 8
    train = _set("ref", M.TRAIN, rng)
    mons = M.build_monitors(train)
    calib = _set("ref", M.CALIB, rng)
    test = _set("ref", M.TEST, rng)
    shifted = _set("w4", M.TEST, rng, shift=0.5)
    for name, mon in mons.items():
        tau = M.conformal_tau([M.episode_score(mon.steps(e)) for e in calib if e.success])
        r = M.evaluate(mon, test, tau)
        assert r["auroc"] > 0.9, (name, r)
        assert r["far"] <= 0.25, (name, r)  # nominal 0.10; small-sample slack
        if name.startswith(("probe", "maha")):
            r2 = M.evaluate(mon, shifted, tau)
            assert r2["far"] > r["far"], (name, r, r2)  # a planted shift must raise FAR


def test_conformal_tau_rank():
    assert M.conformal_tau(list(range(1, 20)), alpha=0.1) == 18.0
    assert M.conformal_tau([1.0, 2.0], alpha=0.1) == float("inf")
