"""Long-format features: one row per (case, unit, hour)."""
import numpy as np
import pandas as pd
import yaml

from data import DATA, H, RESERVOIRS, UNIT_TOPO, load_cases, topology


def _rank(x, axis=-1):
    return x.argsort(axis).argsort(axis).astype(np.float32) / (x.shape[axis] - 1)


def _roll(x, w):
    """Centered rolling mean along last axis with edge padding."""
    pad = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(w // 2, w - 1 - w // 2)], mode="edge")
    c = np.cumsum(np.pad(pad, [(0, 0)] * (x.ndim - 1) + [(1, 0)]), axis=-1)
    return (c[..., w:] - c[..., :-w]) / w


def turbine_qmax():
    """Max discharge (m3/s) per generator = largest flow on its turbine efficiency curves."""
    d = yaml.safe_load(open(DATA / "extended" / "Tokke_Vinje_topology.yaml"))["model"]["generator"]
    out = {}
    for g, v in d.items():
        curves = v["turb_eff_curves"]
        curves = curves if isinstance(curves, list) else [curves]
        out[g] = float(max(max(c["x"]) for c in curves))
    return out


def _segments(b):
    """For boolean (N,H): signed length of the run containing t (+ above / - below), position in it, hours left."""
    N, T = b.shape
    t = np.arange(T)
    change = np.concatenate([np.ones((N, 1), bool), b[:, 1:] != b[:, :-1]], 1)
    start = np.maximum.accumulate(np.where(change, t, 0), axis=1)
    gid = np.cumsum(change, 1) - 1 + (np.arange(N) * T)[:, None]
    length = np.bincount(gid.ravel(), minlength=N * T)[gid]
    pos = t - start
    sign = np.where(b, 1, -1)
    return (sign * length).astype(np.float32), pos.astype(np.float32), (length - pos - 1).astype(np.float32)


def _shift(x, s):
    """x[:, t+s] with edge padding (s>0: future, s<0: past)."""
    idx = np.clip(np.arange(x.shape[1]) + s, 0, x.shape[1] - 1)
    return x[:, idx]


def _window(x, w, forward):
    """Max and min of x over the next (forward) or previous w hours, excluding t; edge padded."""
    shifts = range(1, w + 1) if forward else range(-w, 0)
    st = np.stack([_shift(x, s) for s in shifts])
    return st.max(0), st.min(0)


def build(use_future_vol=True, cross_plant=True, context=False):
    a = load_cases()
    n = len(a["price"])
    units = list(a["units"])
    U = len(units)
    max_vol, gens = topology()
    R = RESERVOIRS
    ri = {r: i for i, r in enumerate(R)}
    vmax = np.array([max_vol[r] for r in R], dtype=np.float32)

    P = a["price"]                                  # n,H
    wv, vol = a["wv"], a["vol"]                     # n,17,8
    infl = a["inflow"]                              # n,18,H
    infl_names = list(a["inflow_names"])
    starts = pd.to_datetime(a["start"])

    # ---------- hour-level (n,H) ----------
    pm, ps = P.mean(1, keepdims=True), P.std(1, keepdims=True) + 1e-3
    Pd = P.reshape(n, 7, 24)
    hour = {
        "price": P,
        "price_z": (P - pm) / ps,
        "price_ratio": P / (np.abs(pm) + 1),
        "price_rank_w": _rank(P),
        "price_rank_d": _rank(Pd).reshape(n, H),
        "price_m_daymean": (Pd - Pd.mean(2, keepdims=True)).reshape(n, H) / ps,
        "price_daymean_z": np.repeat((Pd.mean(2) - pm) / ps, 24, 1),
        "price_daymax_z": np.repeat((Pd.max(2) - pm) / ps, 24, 1),
        "price_daymin_z": np.repeat((Pd.min(2) - pm) / ps, 24, 1),
        "price_prev_z": (np.concatenate([P[:, :1], P[:, :-1]], 1) - P) / ps,
        "price_next_z": (np.concatenate([P[:, 1:], P[:, -1:]], 1) - P) / ps,
    }
    for w in [3, 5, 9, 25]:
        hour[f"price_roll{w}_z"] = (_roll(P, w) - pm) / ps
    # one-sided 24h context: is a pricier period coming / just passed?
    Pp = np.pad(P, ((0, 0), (24, 24)), mode="edge")
    c = np.cumsum(np.pad(Pp, ((0, 0), (1, 0))), 1)
    hour["price_fwd24_z"] = ((c[:, 49:49 + H] - c[:, 25:25 + H]) / 24 - P) / ps
    hour["price_bwd24_z"] = ((c[:, 24:24 + H] - c[:, 0:H]) / 24 - P) / ps
    t = np.broadcast_to(np.arange(H), (n, H))
    hour["t"] = t
    hour["hod"] = t % 24
    hour["day"] = t // 24
    dow0 = starts.dayofweek.to_numpy()[:, None]
    hour["dow"] = (dow0 + t // 24) % 7
    hour["infl_tot_t"] = infl.sum(1)
    for j, name in enumerate(a["minflow_names"]):
        hour[f"minflow_{name}"] = a["minflow"][:, j]
    for j, name in enumerate(a["minvol_names"]):
        hour[f"minvol_{name}"] = a["minvol"][:, j] / max_vol[name]

    # ---------- case-level (n,) ----------
    doy = starts.dayofyear.to_numpy()
    case = {
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "month": starts.month.to_numpy(), "dow0": dow0[:, 0],
        "p_mean": pm[:, 0], "p_std": ps[:, 0], "p_min": P.min(1), "p_max": P.max(1),
        "p_cv": ps[:, 0] / (np.abs(pm[:, 0]) + 1),
    }
    for q in [10, 25, 50, 75, 90]:
        case[f"p_q{q}"] = np.percentile(P, q, axis=1)
    for r in R:
        i = ri[r]
        case[f"vfrac0_{r}"] = vol[:, i, 0] / vmax[i]
        case[f"wv7_{r}"] = wv[:, i, 7]
        case[f"wv7rel_{r}"] = wv[:, i, 7] / (np.abs(pm[:, 0]) + 1)
        case[f"wv0rel_{r}"] = wv[:, i, 0] / (np.abs(pm[:, 0]) + 1)
        if use_future_vol:
            case[f"dvol_{r}"] = (vol[:, i, 7] - vol[:, i, 0]) / vmax[i]
    for j, name in enumerate(infl_names):
        case[f"infl_{name}"] = infl[:, j].mean(1)
    case["infl_total"] = infl.sum(1).mean(1)

    # ---------- unit-level (n,U) and unit x hour (n,U,H) ----------
    uf = {k: np.zeros((n, U), np.float32) for k in [
        "u_pmin", "u_pmax", "u_startcost", "u_gidx", "u_ngen",
        "up_wv0", "up_wv7", "dn_wv0", "dn_wv7", "wvdiff7", "wvdiff0", "wv_trend",
        "up_vfrac0", "dn_vfrac0", "up_infl", "up_infl_rel", "frac_p_gt_wv", "pmean_m_wv",
        "up_dvol", "dn_dvol", "up_dvol_infl", "rank_need_up",
        "frac_up_080", "frac_up_090", "frac_up_100", "frac_up_110", "frac_up_125"]}
    uh = {k: np.zeros((n, U, H), np.float32) for k in [
        "p_m_wvdiff", "p_m_wvdiff_z", "p_over_wvdiff", "up_infl_t", "p_over_upwv", "rank_m_need",
        "rank_m_need_up", "p_over_upwv0", "p_over_upwv_daymax",
        "seg_up_len", "seg_up_pos", "seg_up_rem", "seg_df_len", "seg_df_pos", "seg_df_rem"]
        + ([f"ratio_{d}{s}" for d in ("lag", "lead") for s in (1, 2, 3, 6, 12, 24)]
           + [f"ratio_{d}{w}" for d in ("fmax", "fmin", "bmax", "bmin") for w in (6, 12)] if context else [])}
    rank_w = hour["price_rank_w"]
    plant_units = {}
    for u in units:
        plant_units.setdefault(u.rsplit("_", 1)[0], []).append(u)
    for k, u in enumerate(units):
        plant = u.rsplit("_", 1)[0]
        up, dn = UNIT_TOPO[plant]
        g = gens[u]
        uf["u_pmin"][:, k], uf["u_pmax"][:, k], uf["u_startcost"][:, k] = g["p_min"], g["p_max"], g["startcost"]
        uf["u_gidx"][:, k] = int(u.rsplit("_G", 1)[1])
        uf["u_ngen"][:, k] = len(plant_units[plant])
        iu = [ri[r] for r in up]
        uw0, uw7 = wv[:, iu, 0].mean(1), wv[:, iu, 7].mean(1)
        dw0 = wv[:, ri[dn], 0] if dn else np.zeros(n, np.float32)
        dw7 = wv[:, ri[dn], 7] if dn else np.zeros(n, np.float32)
        uf["up_wv0"][:, k], uf["up_wv7"][:, k], uf["dn_wv0"][:, k], uf["dn_wv7"][:, k] = uw0, uw7, dw0, dw7
        uf["wvdiff7"][:, k], uf["wvdiff0"][:, k] = uw7 - dw7, uw0 - dw0
        uf["wv_trend"][:, k] = uw7 - uw0
        uvm = vmax[iu].sum()
        uf["up_vfrac0"][:, k] = vol[:, iu, 0].sum(1) / uvm
        uf["dn_vfrac0"][:, k] = vol[:, ri[dn], 0] / vmax[ri[dn]] if dn else 0
        ii = [infl_names.index(r) for r in up]
        ui = infl[:, ii].sum(1)                                  # n,H
        uf["up_infl"][:, k] = ui.mean(1)
        uf["up_infl_rel"][:, k] = ui.sum(1) * 3600 / 1e6 / uvm   # weekly inflow as fraction of capacity
        uh["up_infl_t"][:, k] = ui
        thr = (uw7 - dw7)[:, None]
        uh["p_m_wvdiff"][:, k] = P - thr
        uh["p_m_wvdiff_z"][:, k] = (P - thr) / ps
        uh["p_over_wvdiff"][:, k] = P / (np.abs(thr) + 1)
        uf["frac_p_gt_wv"][:, k] = (P > thr).mean(1)
        uh["p_over_upwv"][:, k] = P / (np.abs(uw7[:, None]) + 1)
        # >0 when hour t is among the hours whose price beats the plant's water-value threshold
        uh["rank_m_need"][:, k] = rank_w - (1 - uf["frac_p_gt_wv"][:, k:k + 1])
        uf["pmean_m_wv"][:, k] = (pm[:, 0] - thr[:, 0]) / ps[:, 0]
        # same economics against the upstream water value alone (strongest single signal)
        ratio = P / (np.abs(uw7[:, None]) + 1)
        for c in (0.8, 0.9, 1.0, 1.1, 1.25):
            uf[f"frac_up_{int(c * 100):03d}"][:, k] = (ratio > c).mean(1)
        uf["rank_need_up"][:, k] = 1 - uf["frac_up_100"][:, k]
        uh["rank_m_need_up"][:, k] = rank_w - uf["rank_need_up"][:, k:k + 1]
        uh["p_over_upwv0"][:, k] = P / (np.abs(uw0[:, None]) + 1)
        uh["p_over_upwv_daymax"][:, k] = np.repeat(ratio.reshape(n, 7, 24).max(2), 24, 1)
        # run lengths: start/stop costs make short spikes above (or dips below) the threshold not worth acting on
        (uh["seg_up_len"][:, k], uh["seg_up_pos"][:, k], uh["seg_up_rem"][:, k]) = _segments(ratio > 1)
        (uh["seg_df_len"][:, k], uh["seg_df_pos"][:, k], uh["seg_df_rem"][:, k]) = _segments(P > thr)
        if context:
            for s in (1, 2, 3, 6, 12, 24):
                uh[f"ratio_lag{s}"][:, k] = _shift(ratio, -s)
                uh[f"ratio_lead{s}"][:, k] = _shift(ratio, s)
            for w in (6, 12):
                uh[f"ratio_fmax{w}"][:, k], uh[f"ratio_fmin{w}"][:, k] = _window(ratio, w, forward=True)
                uh[f"ratio_bmax{w}"][:, k], uh[f"ratio_bmin{w}"][:, k] = _window(ratio, w, forward=False)
        if use_future_vol:
            uf["up_dvol"][:, k] = (vol[:, iu, 7].sum(1) - vol[:, iu, 0].sum(1)) / uvm
            uf["dn_dvol"][:, k] = (vol[:, ri[dn], 7] - vol[:, ri[dn], 0]) / vmax[ri[dn]] if dn else 0
            # implied release over the week = inflow - volume change (Mm3)
            uf["up_dvol_infl"][:, k] = ui.sum(1) * 3600 / 1e6 - (vol[:, iu, 7].sum(1) - vol[:, iu, 0].sum(1))
    if not use_future_vol:
        for k in ["up_dvol", "dn_dvol", "up_dvol_infl"]:
            del uf[k]

    # ---------- water balance: how fast reservoirs fill/drain (decisive when prices are flat) ----------
    qmax = turbine_qmax()
    plant_q = {}
    for u in units:
        plant_q[u.rsplit("_", 1)[0]] = plant_q.get(u.rsplit("_", 1)[0], 0.0) + qmax[u]
    feeders = {}                                       # reservoir -> plants discharging into it
    for plant, (_, dn) in UNIT_TOPO.items():
        if dn:
            feeders.setdefault(dn, []).append(plant)
    m3h = 3600 / 1e6                                   # m3/s for one hour -> Mm3
    for name in ["up_fill_t", "up_fill_feed_t", "dn_fill_t", "dn_fill_feed_t", "up_hours_to_full",
                 "up_drain_hours", "dn_room_hours", "feed_q_rel"]:
        shape = (n, U, H) if name.endswith("_t") else (n, U)
        (uh if name.endswith("_t") else uf)[name] = np.zeros(shape, np.float32)
    cum_t = np.arange(1, H + 1, dtype=np.float32)[None, :]
    for k, u in enumerate(units):
        plant = u.rsplit("_", 1)[0]
        up, dn = UNIT_TOPO[plant]
        iu = [ri[r] for r in up]
        v0, vm = vol[:, iu, 0].sum(1), vmax[iu].sum()
        nat = np.cumsum(infl[:, [infl_names.index(r) for r in up]].sum(1), 1) * m3h      # n,H Mm3
        feed_q = sum(plant_q[p] for r in up for p in feeders.get(r, []))
        uh["up_fill_t"][:, k] = (v0[:, None] + nat) / vm
        uh["up_fill_feed_t"][:, k] = (v0[:, None] + nat + feed_q * m3h * cum_t) / vm
        rate = nat[:, -1] / H + 1e-4
        uf["up_hours_to_full"][:, k] = np.clip((vm - v0) / rate, 0, 2000)
        uf["up_drain_hours"][:, k] = np.clip(v0 / (plant_q[plant] * m3h), 0, 5000)
        uf["feed_q_rel"][:, k] = feed_q / plant_q[plant]
        if dn:
            j = ri[dn]
            dnat = np.cumsum(infl[:, infl_names.index(dn)], 1) * m3h
            dfeed = sum(plant_q[p] for p in feeders.get(dn, []))
            uh["dn_fill_t"][:, k] = (vol[:, j, 0][:, None] + dnat) / vmax[j]
            uh["dn_fill_feed_t"][:, k] = (vol[:, j, 0][:, None] + dnat + dfeed * m3h * cum_t) / vmax[j]
            uf["dn_room_hours"][:, k] = np.clip((vmax[j] - vol[:, j, 0]) / (dfeed * m3h), 0, 5000)

    # every plant's economics at hour t, visible to every unit (plants are coupled through the watercourse)
    first_unit = {}
    for k, u in enumerate(units):
        first_unit.setdefault(u.rsplit("_", 1)[0], k)
    for plant, k in (first_unit.items() if cross_plant else []):
        tag = plant.replace(" ", "")
        hour[f"pl_{tag}_ratio"] = uh["p_over_upwv"][:, k]
        hour[f"pl_{tag}_seg"] = uh["seg_up_len"][:, k]
        case[f"pl_{tag}_frac"] = uf["frac_up_100"][:, k]

    # ---------- assemble long frame ----------
    N = n * U * H
    cols, names = [], []
    for k, v in hour.items():
        cols.append(np.broadcast_to(v[:, None, :], (n, U, H)).reshape(N)); names.append(k)
    for k, v in case.items():
        cols.append(np.broadcast_to(np.asarray(v)[:, None, None], (n, U, H)).reshape(N)); names.append(k)
    for k, v in uf.items():
        cols.append(np.broadcast_to(v[:, :, None], (n, U, H)).reshape(N)); names.append(k)
    for k, v in uh.items():
        cols.append(v.reshape(N)); names.append(k)
    cols.append(np.broadcast_to(np.arange(U)[None, :, None], (n, U, H)).reshape(N)); names.append("unit")
    X = np.empty((N, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        X[:, j] = c
    return X, names, a


if __name__ == "__main__":
    import time
    t0 = time.time()
    X, names, a = build()
    print(X.shape, len(names), f"{X.nbytes / 1e9:.2f} GB", f"{time.time() - t0:.1f}s")
