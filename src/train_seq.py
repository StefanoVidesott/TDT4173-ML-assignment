"""Per-(case, unit) sequence model over the long-format features.

The GBM sees one hour at a time; here a dilated TCN sees the unit's whole week (receptive field > 168h)
plus a global-week summary in every block, so it can reason about start/stop costs and the weekly water budget.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cv import folds, micro_auc
from data import ROOT, H
from features import build

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="seq")
ap.add_argument("--folds", default="0,1,2,3")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--d", type=int, default=96)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--full", action="store_true")
args = ap.parse_args()
torch.set_num_threads(8)

X, names, a = build(use_future_vol=False)
n, U = len(a["price"]), len(a["units"])
ui = names.index("unit")
keep = [j for j in range(len(names)) if j != ui]
# robust standardisation in place, column by column (X is ~6 GB)
for j in keep:
    c = X[:, j]
    lo, hi = np.percentile(c[:: 97], [1, 99])
    m, s = c[:: 97].mean(), c[:: 97].std() + 1e-6
    np.clip(c, lo - 3 * (hi - lo + 1e-6), hi + 3 * (hi - lo + 1e-6), out=c)
    c -= m
    c /= s
X = X.reshape(n, U, H, -1)
Xt = torch.from_numpy(X)                     # shares memory
unit_ids = torch.arange(U)
Y = torch.from_numpy(a["y"].astype(np.float32))
keep_t = torch.tensor(keep)


class Block(nn.Module):
    def __init__(self, d, dil, drop):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.ctx = nn.Linear(d, d)
        self.mix = nn.Conv1d(d, d, 1)
        self.norm = nn.GroupNorm(1, d)
        self.drop = nn.Dropout(drop)

    def forward(self, h):                     # B,d,H
        z = self.conv(self.norm(h)) + self.ctx(h.mean(2))[:, :, None]
        return h + self.drop(self.mix(F.gelu(z)))


class TCN(nn.Module):
    def __init__(self, f, d, drop):
        super().__init__()
        self.inp = nn.Linear(f, d)
        self.emb = nn.Embedding(U, d)
        self.blocks = nn.Sequential(*[Block(d, k, drop) for k in (1, 2, 4, 8, 16, 32, 64, 1, 4, 16)])
        self.out = nn.Sequential(nn.GroupNorm(1, d), nn.Conv1d(d, d, 1), nn.GELU(), nn.Conv1d(d, 1, 1))

    def forward(self, x, u):                  # x: B,H,F  u: B
        h = (self.inp(x) + self.emb(u)[:, None]).transpose(1, 2)
        return self.out(self.blocks(h))[:, 0]  # B,H


def batch(idx):
    c, u = idx // U, idx % U
    return Xt[c, u][:, :, keep_t], unit_ids[u], Y[c, u]


def fit_predict(tr_cases, pred_cases, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = TCN(len(keep), args.d, args.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    items = (tr_cases[:, None] * U + np.arange(U)[None]).ravel()
    steps = args.epochs * int(np.ceil(len(items) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = rng.permutation(items)
        for i in range(0, len(perm), args.bs):
            x, u, y = batch(torch.from_numpy(perm[i:i + args.bs]))
            loss = F.binary_cross_entropy_with_logits(model(x, u), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item() * len(u)
        print(f"  ep {ep} loss {tot / len(items):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    items = (pred_cases[:, None] * U + np.arange(U)[None]).ravel()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), 1024):
            x, u, _ = batch(torch.from_numpy(items[i:i + 1024]))
            out.append(torch.sigmoid(model(x, u)))
    return torch.cat(out).numpy().reshape(len(pred_cases), U, H)


out = ROOT / "preds"; out.mkdir(exist_ok=True)
oof = np.full((n, U, H), np.nan, np.float32)
y = a["y"]
for f, tr, va in folds(a["start"], a["is_train"], which=[int(x) for x in args.folds.split(",")]):
    t0 = time.time()
    p = np.mean([fit_predict(tr, va, s) for s in range(args.seeds)], 0)
    oof[va] = p
    print(f"fold {f}: auc {micro_auc(y[va], p):.5f} ({time.time() - t0:.0f}s)", flush=True)
m = a["is_train"] & ~np.isnan(oof[:, 0, 0])
print(f"OOF auc ({m.sum()} cases): {micro_auc(y[m], oof[m]):.5f}")
np.save(out / f"{args.name}_oof.npy", oof)

if args.full:
    tr, te = np.where(a["is_train"])[0], np.where(~a["is_train"])[0]
    pt = np.mean([fit_predict(tr, te, 100 + s) for s in range(args.seeds)], 0)
    np.save(out / f"{args.name}_test.npy", pt.astype(np.float32))
    print("saved test preds", pt.shape)
