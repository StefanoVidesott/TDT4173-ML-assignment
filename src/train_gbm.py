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
ap.add_argument("--context", action="store_true", help="add unit ratio lag/lead/window features (did not help)")
ap.add_argument("--pseudo", default="", help="comma list of prediction names used as soft labels for unlabeled cases")
ap.add_argument("--pseudo-w", type=float, default=1.0, help="sample weight of pseudo-labelled rows")
ap.add_argument("--topo2", action="store_true", help="effective upstream storage features")
ap.add_argument("--dn2", action="store_true", help="with --topo2: effective downstream storage too")
ap.add_argument("--patience", type=int, default=150)
ap.add_argument("--algo", default="lgb", choices=["lgb", "xgb"])
args = ap.parse_args()

X, names, a = build(use_future_vol=not args.no_future_vol, cross_plant=not args.no_cross,
                       context=args.context, topo2=args.topo2, dn2=args.dn2)
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


xparams = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", grow_policy="lossguide",
               max_depth=0, max_leaves=args.leaves, min_child_weight=args.min_leaf / 10, eta=args.lr,
               subsample=0.7, colsample_bytree=args.ff, reg_lambda=10.0, max_bin=127,
               nthread=os.cpu_count(), seed=args.seed)


def fit_xgb(rt, rv, rounds):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(X[rt], y[rt], max_bin=127)
    evals, cb = [], []
    if rv is not None:
        evals = [(xgb.QuantileDMatrix(X[rv], y[rv], ref=dtr, max_bin=127), "va")]
        cb = [xgb.callback.EarlyStopping(args.patience, maximize=True, save_best=True)]
    m = xgb.train(xparams, dtr, rounds, evals=evals, callbacks=cb, verbose_eval=500)
    del dtr
    pred = lambda r: m.predict(xgb.DMatrix(X[r]))
    return pred, (m.best_iteration + 1 if rv is not None else rounds), None


def fit_lgb(rt, rv, rounds, unlabeled_cases):
    dtr = train_set(rt, unlabeled_cases)
    if rv is None:
        m = lgb.train(params, dtr, rounds)
        return (lambda r: m.predict(X[r])), rounds, m
    dva = lgb.Dataset(X[rv], y[rv], reference=dtr)
    m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(args.patience, verbose=False), lgb.log_evaluation(500)],
                  feval=lambda p, d: ("auc", micro_auc(d.get_label(), p), True))
    return (lambda r: m.predict(X[r], num_iteration=m.best_iteration)), m.best_iteration, m


def fit(rt, rv, rounds, unlabeled_cases):
    return fit_xgb(rt, rv, rounds) if args.algo == "xgb" else fit_lgb(rt, rv, rounds, unlabeled_cases)


oof = np.full((n, U, H), np.nan, np.float32)
best_iters = []
for f, tr, va in folds(a["start"], a["is_train"], which=[int(x) for x in args.folds.split(",")]):
    t0 = time.time()
    rt, rv = rows_of(tr, args.sub), rows_of(va)
    pred, it, m = fit(rt, rv, args.rounds, va)
    p = pred(rv)
    oof[va] = p.reshape(len(va), U, H)
    best_iters.append(it)
    print(f"fold {f}: auc {micro_auc(y[rv], p):.5f} iters {it} ({time.time() - t0:.0f}s)", flush=True)
    if f == 3 and m is not None:
        imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)
        print("top features:", [(nm, round(g / 1e3)) for g, nm in imp[:30]])

tr_mask = a["is_train"] & ~np.isnan(oof[:, 0, 0])
print(f"OOF auc ({tr_mask.sum()} cases): {micro_auc(a['y'][tr_mask], oof[tr_mask]):.5f}")
out = Path(os.environ.get("PREDS_DIR", ROOT / "preds")); out.mkdir(parents=True, exist_ok=True)
np.save(out / f"{args.name}_oof.npy", oof)

if args.full:
    te = np.where(~a["is_train"])[0]
    rt = rows_of(np.where(a["is_train"])[0], args.sub)
    pred, _, _ = fit(rt, None, int(np.mean(best_iters) * 1.1), te)
    pt = pred(rows_of(te)).reshape(len(te), U, H).astype(np.float32)
    np.save(out / f"{args.name}_test.npy", pt)
    print("saved test preds", pt.shape)
