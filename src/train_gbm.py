"""LightGBM on long-format features. Saves OOF (train cases) and test predictions as (n_cases, 14, 168)."""
import argparse
import os
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
ap.add_argument("--min-leaf", type=int, default=200)
ap.add_argument("--ff", type=float, default=0.5)
ap.add_argument("--full", action="store_true", help="also fit on all train and predict test")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-cross", action="store_true", help="drop cross-plant features")
ap.add_argument("--pseudo", default="", help="comma list of prediction names used as soft labels for unlabeled cases")
ap.add_argument("--pseudo-w", type=float, default=1.0, help="sample weight of pseudo-labelled rows")
args = ap.parse_args()

X, names, a = build(use_future_vol=not args.no_future_vol, cross_plant=not args.no_cross)
n, U = len(a["price"]), len(a["units"])
y = a["y"].reshape(-1).astype(np.float32)
case_of_row = np.repeat(np.arange(n), U * H)
rng = np.random.default_rng(args.seed)

params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.min_leaf,
              feature_fraction=args.ff, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              max_bin=127, verbose=-1, num_threads=os.cpu_count(), seed=args.seed)
cat = ["unit"]

pseudo = None
if args.pseudo:
    # soft labels for cases whose labels the model may not see: the evaluated fold (simulation) or the test set
    P = Path(os.environ.get("PREDS_DIR", ROOT / "preds"))
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    zs = []
    for nm in args.pseudo.split(","):
        z = np.load(P / f"{nm}_oof.npy")
        z[~a["is_train"]] = np.load(P / f"{nm}_test.npy")
        zs.append(lg(z))
    pseudo = (1 / (1 + np.exp(-np.mean(zs, 0)))).reshape(-1).astype(np.float32)
    params["objective"] = "cross_entropy"


def train_set(rt, unlabeled_cases):
    """Labelled rows rt, plus pseudo-labelled rows of `unlabeled_cases` when --pseudo is set."""
    if pseudo is None:
        return lgb.Dataset(X[rt], y[rt], feature_name=names, categorical_feature=cat, free_raw_data=True)
    rp = rows_of(unlabeled_cases, args.sub)
    w = np.concatenate([np.ones(len(rt), np.float32), np.full(len(rp), args.pseudo_w, np.float32)])
    return lgb.Dataset(np.concatenate([X[rt], X[rp]]), np.concatenate([y[rt], pseudo[rp]]), weight=w,
                       feature_name=names, categorical_feature=cat, free_raw_data=True)


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
    dtr = train_set(rt, va)
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
out = Path(os.environ.get("PREDS_DIR", ROOT / "preds")); out.mkdir(parents=True, exist_ok=True)
np.save(out / f"{args.name}_oof.npy", oof)

if args.full:
    te = np.where(~a["is_train"])[0]
    rt = rows_of(np.where(a["is_train"])[0], args.sub)
    dtr = train_set(rt, te)
    m = lgb.train(params, dtr, int(np.mean(best_iters) * 1.1))
    pt = m.predict(X[rows_of(te)]).reshape(len(te), U, H).astype(np.float32)
    np.save(out / f"{args.name}_test.npy", pt)
    print("saved test preds", pt.shape)
