"""Second stage across units: every unit's prediction sees what all other units are predicted to do.

Units are coupled through the watercourse (Hogga runs on water Tokke releases into Bandak; Vinje G3 only starts
once G1/G2 are already on), but the first-stage GBM predicts each unit independently. The stacker gets, for each
(case, hour), the first-stage logits of all 14 units at t and nearby hours, plus their weekly means.
Trained with the same time-blocked CV on out-of-fold first-stage predictions.
"""
import argparse
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

from cv import folds, micro_auc
from data import ROOT, H, load_cases

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="first-stage prediction names (need _oof.npy, optionally _test.npy)")
ap.add_argument("--name", default="stack2")
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--folds", default="0,1,2,3")
args = ap.parse_args()

a = load_cases()
y, is_tr = a["y"], a["is_train"]
n, U, _ = y.shape
P = Path(os.environ.get("PREDS_DIR", ROOT / "preds"))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
te = np.where(~is_tr)[0]
base = []
for m in args.models:
    b = np.load(P / f"{m}_oof.npy")
    if (P / f"{m}_test.npy").exists():
        b[te] = np.load(P / f"{m}_test.npy")
    base.append(logit(b))
z = np.mean(base, 0)                                            # n,U,H
have = ~np.isnan(z[:, 0, 0])
print("first stage OOF:", round(micro_auc(y[is_tr & have], z[is_tr & have]), 5))


def shift_t(x, s):
    """x[..., t+s] with edge padding."""
    if s == 0:
        return x
    idx = np.clip(np.arange(H) + s, 0, H - 1)
    return x[..., idx]


# features per (case, unit, hour)
feats, names = [], []
def add(arr, name):
    feats.append(np.broadcast_to(arr, (n, U, H)).astype(np.float32)); names.append(name)

for i, b in enumerate(base):
    add(b, f"own_m{i}")
for s in (-6, -3, -1, 1, 3, 6):
    add(shift_t(z, s), f"own_t{s:+d}")
add(z.mean(2, keepdims=True), "own_week_mean")
add(z.std(2, keepdims=True), "own_week_std")
for j in range(U):                                               # every unit's state, seen from every unit
    zj = z[:, j:j + 1, :]
    add(zj, f"u{j}_t")
    add(shift_t(zj, -2), f"u{j}_t-2")
    add(shift_t(zj, 2), f"u{j}_t+2")
    add(zj.mean(2, keepdims=True), f"u{j}_wk")
sig = 1 / (1 + np.exp(-z))
add(sig.sum(1, keepdims=True), "n_units_on_t")
add(np.broadcast_to(np.arange(U)[None, :, None], (n, U, H)), "unit")
add(np.broadcast_to(np.arange(H)[None, None, :], (n, U, H)), "t")
X = np.stack([f.reshape(-1) for f in feats], 1)
del feats
yr = y.reshape(-1).astype(np.float32)
case_of_row = np.repeat(np.arange(n), U * H)
print("stack features:", X.shape)

params = dict(objective="binary", learning_rate=args.lr, num_leaves=127, min_data_in_leaf=300, feature_fraction=0.5,
              bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, max_bin=127, verbose=-1,
              num_threads=os.cpu_count(), seed=0)
rows = lambda c: np.where(np.isin(case_of_row, c))[0]
oof2 = np.full(y.shape, np.nan, np.float32)
iters = []
for f, tr, va in folds(a["start"], is_tr, which=[int(x) for x in args.folds.split(",")]):
    tr = tr[have[tr]]
    rt, rv = rows(tr), rows(va)
    m = lgb.train(params, lgb.Dataset(X[rt], yr[rt], feature_name=names, categorical_feature=["unit"]), 3000,
                  valid_sets=[lgb.Dataset(X[rv], yr[rv])],
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  feval=lambda p, d: ("auc", micro_auc(d.get_label(), p), True))
    oof2[va] = m.predict(X[rv], num_iteration=m.best_iteration).reshape(len(va), U, H)
    iters.append(m.best_iteration)
    print(f"fold {f}: stack {micro_auc(y[va], oof2[va]):.5f}  base {micro_auc(y[va], z[va]):.5f}  iters {m.best_iteration}",
          flush=True)
done = is_tr & ~np.isnan(oof2[:, 0, 0])
print(f"stack OOF {micro_auc(y[done], oof2[done]):.5f}  base {micro_auc(y[done], z[done]):.5f}")
np.save(P / f"{args.name}_oof.npy", oof2)

if have[te].all():
    rt = rows(np.where(is_tr & have)[0])
    m = lgb.train(params, lgb.Dataset(X[rt], yr[rt], feature_name=names, categorical_feature=["unit"]),
                  int(np.mean(iters) * 1.1))
    np.save(P / f"{args.name}_test.npy", m.predict(X[rows(te)]).reshape(len(te), U, H).astype(np.float32))
    print("saved test preds")
