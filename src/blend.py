"""Find logit-blend weights on full OOF, report per-fold scores.

Default: coarse grid on the simplex. With --greedy (needed beyond ~5 models): Caruana ensemble selection,
adding one model at a time with replacement, weights = selection counts / steps."""
import itertools
import sys

import numpy as np

from cv import folds, micro_auc
from data import ROOT, load_cases

greedy = "--greedy" in sys.argv
names = [x for x in sys.argv[1:] if not x.startswith("--")]
a = load_cases()
tr, y = a["is_train"], a["y"]
L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
Z = [L(np.load(ROOT / "preds" / f"{m}_oof.npy"))[tr] for m in names]
yt = y[tr]
F = [np.searchsorted(np.where(tr)[0], v) for _, _, v in folds(a["start"], tr)]
for m, z in zip(names, Z):
    print(f"{m:10s} {micro_auc(yt, z):.5f}")
best = (0, None)
if greedy:
    counts, acc, steps = np.zeros(len(names)), 0.0, 20
    for k in range(1, steps + 1):
        cand = [micro_auc(yt, (acc + z) / k) for z in Z]
        j = int(np.argmax(cand))
        counts[j] += 1; acc = acc + Z[j]
        print(f"step {k}: +{names[j]} {cand[j]:.5f}", flush=True)
    w = counts / steps
    s = micro_auc(yt, sum(wi * z for wi, z in zip(w, Z)))
else:
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
