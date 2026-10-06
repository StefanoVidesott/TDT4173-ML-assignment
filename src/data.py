"""Load raw CSVs and build per-case arrays (one case = 168-hour week starting at `starttime`)."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
H = 168

RESERVOIRS = ["Langesae", "Staavatn", "Kjelavatn", "Bordalsvatn", "Foersvatn", "Hyljelihyl", "Venemo",
              "Bitdalsvatn", "Songavatn", "Totak", "Vaamarvatn", "Langeidvatn", "Vatjern", "Vinjevatn",
              "Botnedalsvatn", "Byrtevatn", "Bandak"]

# unit -> (upstream reservoirs feeding the plant, downstream reservoir or None)
UNIT_TOPO = {
    "Hogga": (["Bandak"], None),
    "Tokke": (["Vinjevatn"], "Bandak"),
    "Lio": (["Byrtevatn"], None),
    "Byrte": (["Botnedalsvatn"], "Byrtevatn"),
    "Vinje": (["Vaamarvatn"], "Vinjevatn"),
    "Haukeli": (["Vatjern"], "Vinjevatn"),
    "Songa": (["Songavatn", "Bitdalsvatn"], "Totak"),
    "Kjela": (["Foersvatn"], "Hyljelihyl"),
    "Vesle Kjela": (["Kjelavatn"], "Foersvatn"),
}


def _read_ts(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date").sort_index()


def unit_columns():
    cols = pd.read_csv(ROOT / "sample_submission.csv", nrows=0).columns[1:]
    units = list(dict.fromkeys("_".join(c.split("_")[2:-1]) for c in cols))
    return list(cols), units


def topology():
    d = yaml.safe_load(open(DATA / "extended" / "Tokke_Vinje_topology.yaml"))["model"]
    max_vol = {r: float(v["max_vol"]) for r, v in d["reservoir"].items()}
    gens = {g: dict(p_min=float(v["p_min"]), p_max=float(v["p_max"]),
                    startcost=float(v.get("startcost_const", 0))) for g, v in d["generator"].items()}
    return max_vol, gens


def load_cases(force=False):
    CACHE.mkdir(exist_ok=True)
    out = CACHE / "cases.npz"
    if out.exists() and not force:
        z = np.load(out, allow_pickle=True)
        return {k: z[k] for k in z.files}

    cols, units = unit_columns()
    uc = pd.read_csv(DATA / "kernel" / "Unit_commitment_decisions.csv")
    test = pd.read_csv(ROOT / "prediction_mapping.csv")
    cases = pd.concat([uc[["Run No", "starttime"]], test]).reset_index(drop=True)
    starts = pd.to_datetime(cases["starttime"]).dt.tz_localize("UTC")
    n = len(cases)

    price = _read_ts(DATA / "kernel" / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0]
    price = price.resample("1h").mean()
    inflow = _read_ts(DATA / "kernel" / "Historical_inflow_1958_2025.csv")
    vol = _read_ts(DATA / "kernel" / "Historical_volume_2015_2024.csv")[RESERVOIRS]
    wv = _read_ts(DATA / "kernel" / "Synthetic_water_value_2015_2024.csv")[RESERVOIRS]
    minvol = _read_ts(DATA / "extended" / "Constraint_min_volume.csv")
    minflow = _read_ts(DATA / "extended" / "Constraint_min_flow.csv")

    def hourly(df, s):
        idx = pd.date_range(s, periods=H, freq="1h")
        return df.reindex(idx).to_numpy(dtype=np.float32)

    def daily(df, s):  # 8 daily points: day 0..7 (day 7 = end of horizon)
        idx = pd.date_range(s, periods=8, freq="1D")
        return df.reindex(idx).to_numpy(dtype=np.float32)

    arr = dict(
        price=np.stack([hourly(price, s) for s in starts]),                       # n,168
        inflow=np.stack([hourly(inflow, s).T for s in starts]),                   # n,18,168
        vol=np.stack([daily(vol, s).T for s in starts]),                          # n,17,8
        wv=np.stack([daily(wv, s).T for s in starts]),                            # n,17,8
        minvol=np.stack([hourly(minvol, s).T for s in starts]),                   # n,3,168
        minflow=np.stack([hourly(minflow, s).T for s in starts]),                 # n,6,168
    )
    y = np.full((n, len(units), H), -1, dtype=np.int8)
    y[: len(uc)] = uc[cols].to_numpy().reshape(len(uc), len(units), H)
    arr.update(
        y=y, run_no=cases["Run No"].to_numpy(), start=starts.dt.strftime("%Y-%m-%d").to_numpy(),
        is_train=np.arange(n) < len(uc), inflow_names=np.array(list(inflow.columns)),
        minvol_names=np.array(list(minvol.columns)), minflow_names=np.array(list(minflow.columns)),
        units=np.array(units),
    )
    for k in ["price", "inflow", "vol", "wv", "minvol", "minflow"]:
        nans = np.isnan(arr[k]).sum()
        if nans:
            print(f"warning: {k} has {nans} NaNs")
    np.savez(out, **arr)
    return arr


if __name__ == "__main__":
    a = load_cases(force=True)
    for k, v in a.items():
        print(k, v.shape, v.dtype)
