"""Write a Kaggle submission from saved test predictions (optionally a logit-mean blend of several models)."""
import argparse

import numpy as np
import pandas as pd

from data import ROOT, load_cases

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="name[:weight]")
ap.add_argument("--out", default="submissions/submission.csv")
args = ap.parse_args()

a = load_cases()
te = np.where(~a["is_train"])[0]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
z, wsum = 0.0, 0.0
for spec in args.models:
    name, w = (spec.split(":") + ["1"])[:2]
    z = z + float(w) * logit(np.load(ROOT / "preds" / f"{name}_test.npy"))
    wsum += float(w)
p = 1 / (1 + np.exp(-z / wsum))

sample = pd.read_csv(ROOT / "sample_submission.csv")
sub = pd.DataFrame(p.reshape(len(te), -1), columns=sample.columns[1:])
sub.insert(0, "Run No", a["run_no"][te])
assert list(sub.columns) == list(sample.columns) and (sub["Run No"].to_numpy() == sample["Run No"].to_numpy()).all()
assert np.isfinite(sub.iloc[:, 1:].to_numpy()).all()
out = ROOT / args.out
out.parent.mkdir(exist_ok=True)
sub.to_csv(out, index=False, float_format="%.6f")
print("wrote", out, sub.shape)
