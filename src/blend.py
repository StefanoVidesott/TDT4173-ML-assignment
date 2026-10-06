"""Find logit-blend weights on full OOF (coarse grid on the simplex), report per-fold scores."""
import itertools
import sys

import numpy as np

from cv import folds, micro_auc
from data import ROOT, load_cases

names = sys.argv[1:]
a = load_cases()
tr, y = a["is_train"], a["y"]
L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
Z = [L(np.load(ROOT / "preds" / f"{m}_oof.npy"))[tr] for m in names]
yt = y[tr]
F = [np.searchsorted(np.where(tr)[0], v) for _, _, v in folds(a["start"], tr)]
for m, z in zip(names, Z):
    print(f"{m:10s} {micro_auc(yt, z):.5f}")
best = (0, None)
grid = np.arange(0, 1.01, 0.1)
for w in itertools.product(grid, repeat=len(names)):
    if abs(sum(w) - 1) > 1e-6:
        continue
    s = micro_auc(yt, sum(wi * z for wi, z in zip(w, Z)))
    if s > best[0]:
        best = (s, w)
s, w = best
b = sum(wi * z for wi, z in zip(w, Z))
print("best weights", dict(zip(names, np.round(w, 2))), f"OOF {s:.5f}", "folds", [round(micro_auc(yt[f], b[f]), 5) for f in F])
eq = sum(Z) / len(Z)
print(f"equal weights OOF {micro_auc(yt, eq):.5f}", "folds", [round(micro_auc(yt[f], eq[f]), 5) for f in F])
