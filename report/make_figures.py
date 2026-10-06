"""Generate the report figures from the data and saved out-of-fold predictions.

Run from the project root:  .venv/bin/python report/make_figures.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cv import folds, micro_auc  # noqa: E402
from data import RESERVOIRS, UNIT_TOPO, load_cases  # noqa: E402
from features import build  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 150})
BLUE, ORANGE, GREY, GREEN = "#3366aa", "#dd8833", "#999999", "#55aa66"

a = load_cases()
tr = a["is_train"]
y = a["y"]
units = [u.replace("_", " ") for u in a["units"]]
L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
W = {"gbm_v2": .4, "gbm_v3": .1, "gbm_v4": .3, "gbm_v5": .2}
blend = sum(w * L(np.load(ROOT / "preds" / f"{m}_oof.npy")) for m, w in W.items())

# 1) EDA: weekly behaviour of each unit (all-off / mixed / all-on)
fr = y[tr].mean(2)
off, on = (fr == 0).mean(0), (fr == 1).mean(0)
mixed = 1 - off - on
fig, ax = plt.subplots(figsize=(6.4, 2.6))
x = np.arange(len(units))
ax.bar(x, off, color=GREY, label="off all week")
ax.bar(x, mixed, bottom=off, color=ORANGE, label="mixed")
ax.bar(x, on, bottom=off + mixed, color=BLUE, label="on all week")
ax.set_xticks(x, units, rotation=45, ha="right")
ax.set_ylabel("share of training cases")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
fig.savefig(OUT / "eda_units.pdf")
plt.close(fig)

# 2) Example weeks: price / upstream water value vs. commitment
def example(ax, unit, start):
    c = np.where(a["start"] == start)[0][0]
    k = list(a["units"]).index(unit)
    up = UNIT_TOPO[unit.rsplit("_", 1)[0]][0]
    wv = a["wv"][c, [RESERVOIRS.index(r) for r in up], 7].mean()
    ratio = a["price"][c] / wv
    t = np.arange(168)
    ax.fill_between(t, 0, y[c, k] * max(ratio.max(), 1.2) * 1.05, step="post", color=BLUE, alpha=.18,
                    label="unit on (ground truth)")
    ax.plot(t, ratio, color=BLUE, lw=1, label="price / upstream water value")
    ax.plot(t, 1 / (1 + np.exp(-blend[c, k])) * max(ratio.max(), 1.2), color=ORANGE, lw=1, ls="--",
            label="predicted P(on) (scaled)")
    ax.axhline(1, color=GREY, lw=.8, ls=":")
    ax.set_title(f"{unit.replace('_', ' ')}, week starting {start}", fontsize=9)
    ax.set_xticks(range(0, 169, 24))
    ax.set_xlabel("hour of the scheduling horizon")


fig, axs = plt.subplots(1, 2, figsize=(7, 2.5), sharey=False)
example(axs[0], "Tokke_G2", "2021-03-03")
example(axs[1], "Hogga_G1", "2021-06-10")
axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.38), ncol=3, frameon=False)
fig.savefig(OUT / "example_weeks.pdf")
plt.close(fig)

# 3) Per-unit and per-year OOF AUC of the final blend
yt, zt = y[tr], blend[tr]
ua = [micro_auc(yt[:, k], zt[:, k]) for k in range(len(units))]
years = pd.to_datetime(a["start"][tr]).year.to_numpy()
ya = {yr: micro_auc(yt[years == yr], zt[years == yr]) for yr in np.unique(years)}
fig, axs = plt.subplots(1, 2, figsize=(7, 2.4), gridspec_kw={"width_ratios": [1.7, 1]})
axs[0].bar(x, ua, color=BLUE)
axs[0].set_xticks(x, units, rotation=45, ha="right")
axs[0].set_ylim(0.93, 1.0)
axs[0].set_ylabel("OOF ROC-AUC")
axs[0].set_title("per unit", fontsize=9)
axs[1].bar([str(k) for k in ya], list(ya.values()), color=[ORANGE if v < .97 else BLUE for v in ya.values()])
axs[1].set_ylim(0.93, 1.0)
axs[1].tick_params(axis="x", rotation=45)
axs[1].set_title("per year", fontsize=9)
fig.savefig(OUT / "auc_breakdown.pdf")
plt.close(fig)

# 4) Learning progress: CV (OOF) vs Kaggle public score
steps = ["unit prior", "GBM v1", "GBM v2", "blend v2+v4", "blend v2-v5"]
cv = [0.6167, 0.9716, 0.9826, 0.9842, 0.9844]
kag = [np.nan, np.nan, 0.98881, 0.99045, 0.99043]
fig, ax = plt.subplots(figsize=(4.6, 2.4))
xs = np.arange(1, len(steps))
ax.plot(xs, cv[1:], "o-", color=BLUE, label="time-blocked CV (OOF)")
ax.plot(xs, kag[1:], "s-", color=ORANGE, label="Kaggle public LB")
ax.set_xticks(xs, steps[1:], rotation=20)
ax.set_ylabel("micro ROC-AUC")
ax.legend(frameon=False)
fig.savefig(OUT / "progress.pdf")
plt.close(fig)

# 5) Feature importance (gain) of the final feature set, fold 3 model
X, names, _ = build(use_future_vol=False, cross_plant=True)
n, U, H = y.shape
case_of_row = np.repeat(np.arange(n), U * H)
f3 = list(folds(a["start"], tr, which=[3]))[0]
rng = np.random.default_rng(0)
rt = np.where(np.isin(case_of_row, f3[1]))[0]
rt = rt[rng.random(len(rt)) < 0.15]
m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.5,
                   bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=127, verbose=-1, seed=0),
              lgb.Dataset(X[rt], y.reshape(-1)[rt].astype(np.float32), feature_name=names,
                          categorical_feature=["unit"]), 250)
gain = pd.Series(m.feature_importance("gain"), index=names).sort_values(ascending=False)
top = gain.head(20)[::-1] / gain.sum()
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.barh(range(len(top)), top.values, color=BLUE)
ax.set_yticks(range(len(top)), list(top.index), fontsize=7)
ax.set_xlabel("share of total split gain")
fig.savefig(OUT / "importance.pdf")
plt.close(fig)
gain.to_csv(OUT / "importance.csv")

print("per-unit AUC:", dict(zip(units, np.round(ua, 4))))
print("per-year AUC:", {k: round(v, 4) for k, v in ya.items()})
print("blend OOF:", round(micro_auc(yt, zt), 5))
print("top-10 features:", list(gain.head(10).index))
