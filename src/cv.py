"""Time-blocked CV folds with a purge gap (consecutive cases overlap 6 of 7 days)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FOLDS = [(2015, 2016), (2017, 2018), (2019, 2020), (2021, 2022)]
GAP_DAYS = 7


def folds(start, is_train, which=None):
    """Yield (fold_idx, train_case_idx, val_case_idx)."""
    d = pd.to_datetime(start)
    tr_all = np.where(is_train)[0]
    for f, (y0, y1) in enumerate(FOLDS):
        if which is not None and f not in which:
            continue
        lo, hi = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y1}-12-31")
        val = tr_all[(d[tr_all] >= lo) & (d[tr_all] <= hi)]
        far = (d[tr_all] < lo - pd.Timedelta(days=GAP_DAYS)) | (d[tr_all] > hi + pd.Timedelta(days=GAP_DAYS))
        yield f, tr_all[far], val


def micro_auc(y, p):
    return roc_auc_score(np.asarray(y).ravel(), np.asarray(p).ravel())
