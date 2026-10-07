"""Week-level model: one row per (case, unit), target = share of on-hours in the week.

With the true weekly share, re-levelling the hourly blend gives 0.997 AUC, so the week level is where the error is.
This model predicts the share from weekly aggregates of the hourly features; `relevel` shifts each (case, unit)
row of hourly logits so that its mean probability matches a mix of the hourly and the weekly estimate.
Saves (n, U) predicted shares as preds/<name>_week_{oof,test}.npy.
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np

from cv import folds, micro_auc
from data import ROOT, H
from features import build

QS = (0, 5, 10, 25, 50, 75, 90, 95, 100)
SORTED = ("p_over_upwv", "p_over_wvdiff", "p_m_wvdiff_z", "up_fill_t", "dn_fill_t", "price_z")


def week_features(cross_plant=False):
    X, names, a = build(use_future_vol=False, cross_plant=cross_plant)
    n, U = len(a["price"]), len(a["units"])
    X4 = X.reshape(n, U, H, len(names))
    cols, fn = [], []
    hourly = [j for j in range(len(names)) if X4[:50, :, :, j].std(2).max() > 0]
    static = [j for j in range(len(names)) if j not in hourly]
    cols.append(X4[:, :, 0, static]); fn += names_of(names, static, "")
    # column by column: fancy-indexing all hourly columns at once would copy the whole 6 GB table
    cols.append(np.stack([X4[:, :, :, j].mean(2) for j in hourly], -1)); fn += names_of(names, hourly, "mean_")
    cols.append(np.stack([X4[:, :, :, j].std(2) for j in hourly], -1)); fn += names_of(names, hourly, "std_")
    for nm in SORTED:
        q = np.percentile(X4[:, :, :, names.index(nm)], QS, axis=2)       # Q,n,U
        cols.append(np.moveaxis(q, 0, -1)); fn += [f"q{p}_{nm}" for p in QS]
    r = X4[:, :, :, names.index("p_over_upwv")]
    for c in (0.7, 0.75, 0.85, 0.95, 1.05, 1.15, 1.4, 1.6):
        cols.append((r > c).mean(2)[..., None]); fn.append(f"frac_up_{c}")
    # siblings in the watercourse: every unit sees every unit's key weekly economics
    key = np.concatenate([(r > 1).mean(2), r.mean(2)], 1)                  # n, 2U
    cols.append(np.broadcast_to(key[:, None, :], (n, U, 2 * U))); fn += [f"all_{i}" for i in range(2 * U)]
    del X, X4
    W = np.concatenate([c.astype(np.float32) for c in cols], -1)
    return W, fn, a


def names_of(names, idx, prefix):
    return [prefix + names[j] for j in idx]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wk")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cache = ROOT / "cache" / "week_features.npz"
    if cache.exists():
        z = np.load(cache)
        Xw, fn = z["W"], list(z["fn"])
        from data import load_cases
        a = load_cases()
    else:
        Xw, fn, a = week_features()
        np.savez(cache, W=Xw, fn=np.array(fn))
    n, U = Xw.shape[:2]
    y = a["y"].astype(np.float32).mean(2)                                  # n,U share of on-hours
    print("week features", Xw.shape, flush=True)
    params = dict(objective="cross_entropy", learning_rate=args.lr, num_leaves=args.leaves,
                  min_data_in_leaf=args.min_leaf, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())
    flat = lambda c: Xw[c].reshape(-1, Xw.shape[-1])
    oof = np.full((n, U), np.nan, np.float32)
    iters = []
    for f, tr, va in folds(a["start"], a["is_train"]):
        t0 = time.time()
        ps = []
        for s in range(args.seeds):
            dtr = lgb.Dataset(flat(tr), y[tr].ravel(), feature_name=fn, categorical_feature=["unit"])
            dva = lgb.Dataset(flat(va), y[va].ravel(), reference=dtr)
            m = lgb.train({**params, "seed": s}, dtr, 5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            ps.append(m.predict(flat(va), num_iteration=m.best_iteration)); iters.append(m.best_iteration)
        oof[va] = np.mean(ps, 0).reshape(len(va), U)
        print(f"fold {f}: MAE {np.abs(oof[va] - y[va]).mean():.4f} iters {iters[-args.seeds:]} "
              f"({time.time() - t0:.0f}s)", flush=True)
    tr = a["is_train"]
    print(f"OOF MAE {np.abs(oof[tr] - y[tr]).mean():.4f}")
    imp = sorted(zip(m.feature_importance("gain"), fn), reverse=True)[:25]
    print("top:", [(k, round(g)) for g, k in imp])
    out = ROOT / "preds"
    np.save(out / f"{args.name}_week_oof.npy", oof)
    te = np.where(~tr)[0]
    trn = np.where(tr)[0]
    pt = []
    for s in range(args.seeds):
        m = lgb.train({**params, "seed": s}, lgb.Dataset(flat(trn), y[trn].ravel(), feature_name=fn,
                                                         categorical_feature=["unit"]), int(np.mean(iters) * 1.1))
        pt.append(m.predict(flat(te)))
    np.save(out / f"{args.name}_week_test.npy", np.mean(pt, 0).reshape(len(te), U).astype(np.float32))
    print("saved")
