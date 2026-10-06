"""Second stage: blend base models + use neighbouring cases' predictions for the same absolute hour.

Case d hour t is the same wall-clock hour as case d+k hour t-24k. Base-model predictions of those
neighbours are fed as features to a LightGBM stacker trained with the same time-blocked CV.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv import folds, micro_auc
from data import ROOT, H, load_cases

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--out", default="submission.csv")
args = ap.parse_args()

a = load_cases()
y, is_tr = a["y"], a["is_train"]
n, U, _ = y.shape
P = ROOT / "preds"
oofs = [np.load(P / f"{m}_oof.npy") for m in args.models]
tests = [np.load(P / f"{m}_test.npy") if (P / f"{m}_test.npy").exists() else None for m in args.models]
te = np.where(~is_tr)[0]

for m, o in zip(args.models, oofs):
    print(f"{m:20s} OOF {micro_auc(y[is_tr], o[is_tr]):.5f}")

# full-series base predictions: OOF for train cases, test preds for test cases
have_test = all(t is not None for t in tests)
base = []
for o, t in zip(oofs, tests):
    b = o.copy()
    if t is not None:
        b[te] = t
    base.append(b)
logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
avg = np.mean([logit(b) for b in base], 0)
print(f"{'mean-logit blend':20s} OOF {micro_auc(y[is_tr], avg[is_tr]):.5f}")


def shifted(x, k):
    """value of case d+k at the same wall-clock hour as (d, t); NaN when outside horizon/data."""
    out = np.full_like(x, np.nan)
    if k > 0:   # later case: (d+k, t-24k)
        out[:-k, :, 24 * k:] = x[k:, :, : H - 24 * k]
    else:       # earlier case: (d-k, t+24k)
        k = -k
        out[k:, :, : H - 24 * k] = x[:-k, :, 24 * k:]
    return out


fold_id = np.full(n, -1)
for f, _, va in folds(a["start"], is_tr):
    fold_id[va] = f
feats = {f"m{i}": logit(b) for i, b in enumerate(base)}
feats["avg"] = avg
for k in range(1, args.k + 1):
    for s, kk in (("+", k), ("-", -k)):
        v = shifted(avg, kk)
        # neighbour from another CV fold was predicted by a model that saw this case's label -> drop it
        nb_fold = np.roll(fold_id, -kk)
        valid = nb_fold == fold_id
        valid[(np.arange(n) + kk < 0) | (np.arange(n) + kk >= n)] = False
        v[~valid] = np.nan
        feats[f"nb{s}{k}"] = v
nbs = np.stack([feats[f"nb{s}{k}"] for k in range(1, args.k + 1) for s in "+-"], 0)
feats["nb_mean"] = np.nanmean(np.concatenate([nbs, avg[None]]), 0)
feats["nb_cnt"] = np.isfinite(nbs).sum(0).astype(np.float32)
# temporal context within the case
feats["prev_h"] = np.concatenate([avg[:, :, :1], avg[:, :, :-1]], 2)
feats["next_h"] = np.concatenate([avg[:, :, 1:], avg[:, :, -1:]], 2)
feats["case_unit_mean"] = np.broadcast_to(avg.mean(2, keepdims=True), avg.shape)
feats["t"] = np.broadcast_to(np.arange(H)[None, None], avg.shape)
feats["unit"] = np.broadcast_to(np.arange(U)[None, :, None], avg.shape)
names = list(feats)
X = np.stack([feats[k].reshape(-1) for k in names], 1).astype(np.float32)
yr = y.reshape(-1).astype(np.float32)
case_of_row = np.repeat(np.arange(n), U * H)
print(f"{'neighbour mean':20s} OOF {micro_auc(y[is_tr], feats['nb_mean'][is_tr]):.5f}")

params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.5, bagging_freq=1, verbose=-1, num_threads=8, seed=0)
rows = lambda c: np.where(np.isin(case_of_row, c))[0]
oof2 = np.full(y.shape, np.nan, np.float32)
iters = []
for f, tr, va in folds(a["start"], is_tr):
    rt, rv = rows(tr), rows(va)
    m = lgb.train(params, lgb.Dataset(X[rt], yr[rt], feature_name=names, categorical_feature=["unit"]), 2000,
                  valid_sets=[lgb.Dataset(X[rv], yr[rv])], callbacks=[lgb.early_stopping(100, verbose=False)],
                  feval=lambda p, d: ("auc", micro_auc(d.get_label(), p), True))
    oof2[va] = m.predict(X[rv], num_iteration=m.best_iteration).reshape(len(va), U, H)
    iters.append(m.best_iteration)
    print(f"stack fold {f}: {micro_auc(y[va], oof2[va]):.5f} (base {micro_auc(y[va], avg[va]):.5f})", flush=True)
print(f"{'stacked':20s} OOF {micro_auc(y[is_tr], oof2[is_tr]):.5f}")

if have_test:
    rt = rows(np.where(is_tr)[0])
    m = lgb.train(params, lgb.Dataset(X[rt], yr[rt], feature_name=names, categorical_feature=["unit"]),
                  int(np.mean(iters) * 1.1))
    pt = m.predict(X[rows(te)]).reshape(len(te), U, H)
    cols = pd.read_csv(ROOT / "sample_submission.csv", nrows=0).columns
    sub = pd.DataFrame(pt.reshape(len(te), -1), columns=cols[1:])
    sub.insert(0, "Run No", a["run_no"][te])
    sub.to_csv(ROOT / args.out, index=False, float_format="%.5f")
    print("wrote", args.out, sub.shape)
