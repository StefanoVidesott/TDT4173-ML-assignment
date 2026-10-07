"""Shift each (case, unit) row of hourly logits so its mean probability hits a target weekly share."""
import numpy as np


def shift_rows(z, target, iters=40):
    """z: (..., H) logits, target: (...) share in (0,1). Bisection on a per-row additive shift."""
    lo, hi = np.full(target.shape, -30.0), np.full(target.shape, 30.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = (1 / (1 + np.exp(-(z + mid[..., None])))).mean(-1)
        lo, hi = np.where(m < target, mid, lo), np.where(m < target, hi, mid)
    return z + ((lo + hi) / 2)[..., None]


def relevel(z, week_p, beta):
    """Mix hourly-implied share and weekly-model share in logit space with weight beta, then re-level z."""
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    ph = (1 / (1 + np.exp(-z))).mean(-1)
    t = 1 / (1 + np.exp(-((1 - beta) * lg(ph) + beta * lg(week_p))))
    return shift_rows(z, t)
