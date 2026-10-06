"""Sequence NN: per case, 168-step input sequence + static context -> 14 x 168 logits."""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cv import folds, micro_auc
from data import ROOT, H, RESERVOIRS, UNIT_TOPO, load_cases, topology

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="nn")
ap.add_argument("--folds", default="0,1,2,3")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--arch", default="tf", choices=["tf", "gru"])
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--full", action="store_true")
args = ap.parse_args()
torch.set_num_threads(8)


def make_inputs(a):
    n = len(a["price"])
    units = list(a["units"])
    max_vol, gens = topology()
    R = RESERVOIRS
    ri = {r: i for i, r in enumerate(R)}
    vmax = np.array([max_vol[r] for r in R], np.float32)
    P, wv, vol = a["price"], a["wv"], a["vol"]
    pm = np.abs(P.mean(1, keepdims=True)) + 1
    ps = P.std(1, keepdims=True) + 1e-3
    t = np.arange(H)
    starts = pd.to_datetime(a["start"])
    dow0 = starts.dayofweek.to_numpy()[:, None]

    seq = [P / pm, (P - P.mean(1, keepdims=True)) / ps, P.argsort(1).argsort(1) / (H - 1),
           np.broadcast_to(np.sin(2 * np.pi * t / 24), (n, H)), np.broadcast_to(np.cos(2 * np.pi * t / 24), (n, H)),
           np.broadcast_to(t / H, (n, H)), ((dow0 + t // 24) % 7 >= 5).astype(np.float32)]
    seq += list(np.log1p(np.clip(a["inflow"], 0, None)).transpose(1, 0, 2))
    seq += list(np.log1p(a["minflow"]).transpose(1, 0, 2))
    seq += list((a["minvol"] / np.array([max_vol[r] for r in a["minvol_names"]])[None, :, None]).transpose(1, 0, 2))
    # per-unit economic signal: price relative to the water value difference across the plant
    for u in units:
        up, dn = UNIT_TOPO[u.rsplit("_", 1)[0]]
        thr = wv[:, [ri[r] for r in up], 7].mean(1) - (wv[:, ri[dn], 7] if dn else 0)
        seq.append((P - thr[:, None]) / ps)
        seq.append(P / (np.abs(thr[:, None]) + 1))
    seq = np.stack(seq, -1).astype(np.float32)                     # n,H,C

    doy = starts.dayofyear.to_numpy()
    stat = [np.log(pm[:, 0]), ps[:, 0] / pm[:, 0], np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    stat += list((wv[:, :, 7] / pm).T) + list((wv[:, :, 0] / pm).T) + list(np.log1p(np.clip(wv[:, :, 7], 0, None)).T)
    stat += list((vol[:, :, 0] / vmax).T)
    stat += list(np.log1p(np.clip(a["inflow"], 0, None).mean(2)).T)
    stat = np.stack(stat, -1).astype(np.float32)                   # n,S

    def standardize(x, axes):
        m = x.mean(axes, keepdims=True); s = x.std(axes, keepdims=True) + 1e-6
        return (x - m) / s
    return standardize(seq, (0, 1)), standardize(stat, (0,))


class Net(nn.Module):
    def __init__(self, c, s, d, layers, drop, n_units=14):
        super().__init__()
        self.inp = nn.Linear(c, d)
        self.stat = nn.Sequential(nn.Linear(s, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, H, d) * 0.02)
        if args.arch == "tf":
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers)
        else:
            self.enc = nn.GRU(d, d // 2, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.conv = nn.Conv1d(d, d, 5, padding=2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n_units))

    def forward(self, x, s):
        h = self.inp(x) + self.stat(s)[:, None] + self.pos
        h = self.enc(h) if args.arch == "tf" else self.enc(h)[0]
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.head(h).transpose(1, 2)                      # B,U,H


def fit_predict(seq, stat, y, tr, va_list, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(seq.shape[-1], stat.shape[-1], args.d, args.layers, args.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    bs = 32
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    X, S, Y = torch.tensor(seq), torch.tensor(stat), torch.tensor(y.astype(np.float32))
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(model(X[b], S[b]), Y[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    outs = []
    with torch.no_grad():
        for va in va_list:
            outs.append(torch.cat([torch.sigmoid(model(X[va[i:i + 256]], S[va[i:i + 256]]))
                                   for i in range(0, len(va), 256)]).numpy())
    return outs


a = load_cases()
seq, stat = make_inputs(a)
y = a["y"]
n = len(y)
oof = np.full(y.shape, np.nan, np.float32)
out = ROOT / "preds"; out.mkdir(exist_ok=True)
for f, tr, va in folds(a["start"], a["is_train"], which=[int(x) for x in args.folds.split(",")]):
    t0 = time.time()
    p = np.mean([fit_predict(seq, stat, y, tr, [va], args.epochs, s)[0] for s in range(args.seeds)], 0)
    oof[va] = p
    print(f"fold {f}: auc {micro_auc(y[va], p):.5f} ({time.time() - t0:.0f}s)", flush=True)
m = a["is_train"] & ~np.isnan(oof[:, 0, 0])
print(f"OOF auc ({m.sum()} cases): {micro_auc(y[m], oof[m]):.5f}")
np.save(out / f"{args.name}_oof.npy", oof)

if args.full:
    tr, te = np.where(a["is_train"])[0], np.where(~a["is_train"])[0]
    pt = np.mean([fit_predict(seq, stat, y, tr, [te], args.epochs, 100 + s)[0] for s in range(args.seeds)], 0)
    np.save(out / f"{args.name}_test.npy", pt.astype(np.float32))
    print("saved test preds", pt.shape)
