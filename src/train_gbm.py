"""LightGBM on long-format features. Saves OOF (train cases) and test predictions as (n_cases, 14, 168)."""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from cv import folds, micro_auc
from data import ROOT, H
from features import build

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="gbm")
ap.add_argument("--folds", default="0,1,2,3")
ap.add_argument("--no-future-vol", action="store_true")
ap.add_argument("--sub", type=float, default=0.35, help="row subsample of training data")
ap.add_argument("--rounds", type=int, default=3000)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--leaves", type=int, default=255)
ap.add_argument("--full", action="store_true", help="also fit on all train and predict test")
args = ap.parse_args()

X, names, a = build(use_future_vol=not args.no_future_vol)
n, U = len(a["price"]), len(a["units"])
y = a["y"].reshape(-1).astype(np.float32)
case_of_row = np.repeat(np.arange(n), U * H)
rng = np.random.default_rng(0)

params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              max_bin=127, verbose=-1, num_threads=8, seed=0)
cat = ["unit"]


def rows_of(cases, frac=1.0):
    mask = np.zeros(n, bool); mask[cases] = True
    r = np.where(mask[case_of_row])[0]
    if frac < 1:
        r = r[rng.random(len(r)) < frac]
    return r


oof = np.full((n, U, H), np.nan, np.float32)
best_iters = []
for f, tr, va in folds(a["start"], a["is_train"], which=[int(x) for x in args.folds.split(",")]):
    t0 = time.time()
    rt, rv = rows_of(tr, args.sub), rows_of(va)
    dtr = lgb.Dataset(X[rt], y[rt], feature_name=names, categorical_feature=cat, free_raw_data=True)
    dva = lgb.Dataset(X[rv], y[rv], reference=dtr)
    m = lgb.train(params, dtr, args.rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
                  feval=lambda p, d: ("auc", micro_auc(d.get_label(), p), True))
    p = m.predict(X[rv], num_iteration=m.best_iteration)
    oof[va] = p.reshape(len(va), U, H)
    best_iters.append(m.best_iteration)
    print(f"fold {f}: auc {micro_auc(y[rv], p):.5f} iters {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    if f == 3:
        imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)
        print("top features:", [(nm, round(g / 1e3)) for g, nm in imp[:30]])

tr_mask = a["is_train"] & ~np.isnan(oof[:, 0, 0])
print(f"OOF auc ({tr_mask.sum()} cases): {micro_auc(a['y'][tr_mask], oof[tr_mask]):.5f}")
out = ROOT / "preds"; out.mkdir(exist_ok=True)
np.save(out / f"{args.name}_oof.npy", oof)

if args.full:
    te = np.where(~a["is_train"])[0]
    rt = rows_of(np.where(a["is_train"])[0], args.sub)
    dtr = lgb.Dataset(X[rt], y[rt], feature_name=names, categorical_feature=cat)
    m = lgb.train(params, dtr, int(np.mean(best_iters) * 1.1))
    pt = m.predict(X[rows_of(te)]).reshape(len(te), U, H).astype(np.float32)
    np.save(out / f"{args.name}_test.npy", pt)
    print("saved test preds", pt.shape)
