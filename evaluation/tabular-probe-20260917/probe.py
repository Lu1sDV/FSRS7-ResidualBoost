#!/usr/bin/env python
"""Tabular probe: does an event-level feature set beat B on the lockbox panel?

B is the published equal-user correction on top of the personalized FSRS-7 logit
(`z_F`): logit p_B = z_F + beta0 + beta16*x16 + beta256*x256.  This probe keeps
`logit(clip(p_B))` as the *offset* (XGBoost `base_margin`, i.e. the model starts
at B) and asks whether strictly causal per-review features add anything.

    .venv/bin/python evaluation/tabular-probe-20260917/probe.py build --procs 6
    .venv/bin/python evaluation/tabular-probe-20260917/probe.py run
    .venv/bin/python evaluation/tabular-probe-20260917/probe.py check

Phases:
  build  -- per-review features -> features-all.parquet (+ provenance checks)
  run    -- user-split XGBoost / logistic ablation -> results.json, gate-*.json
  check  -- causality (perturb rows >= t, features for t unchanged) + provenance

Frame source (see README): `evaluation/full-benchmark-9999-20260916/predictions/
<user>.npz` -- the per-user frames the harness actually scored.  It carries
y / p^F (official FSRS-7) / i / elapsed_days / lapse / fold but no card or deck
ids, so the scored row order (fold-major concat of the five equalized test
folds) is reconstructed from the raw revlogs with the harness's own feature
engineer and verified row-for-row against the npz before `card_id`/`deck_id`/
`note_id` are joined from `data/cards`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPINE = ROOT / "evaluation/mixture-ab-20260916/panel2/aligned-panel2.parquet"
NPZ_DIR = ROOT / "evaluation/full-benchmark-9999-20260916/predictions"
FEAT_MM = ROOT / "evaluation/crossfit-equal-user-20260916/features"
PANEL_JSON = ROOT / "srs-benchmark/result/lockbox-panel-256-seed20260817-users1-4999.json"
GATE = ROOT / "evaluation/promotion-baseline-20260917/gate_report.py"
CROSSFIT = ROOT / "evaluation/mixture-ab-20260916/sources/crossfit-eq/crossfit.json"
FRESH_DIR = ROOT / "evaluation/fresh-panel-20260917"
FRESH_USERS = FRESH_DIR / "users.json"
FRESH_REF = FRESH_DIR / "references.json"
FRESH_PARQUET = HERE / "features-fresh.parquet"
DATA = ROOT / "data"
OUT_PARQUET = HERE / "features-all.parquet"
SHRINK = 64          # n/(n+64) shrinkage, frozen by residual-decomposition-20260909
SURPRISE_WINDOWS = (16, 64)
CLIP = 1e-6          # logit clip for p_B / p^F

# ---------------------------------------------------------------- feature groups

F_SPINE = ["x16", "x256"]
F_SURPRISE = [f"{s}_{stat}{w}" for w in SURPRISE_WINDOWS for s in ("sF", "sB")
              for stat in ("mean", "max")] + ["sF_last", "sB_last", "sF_all", "sB_all"]
F_STATE = ["st_i", "st_elapsed_days", "st_lapse", "st_t", "st_card_seq",
           "st_age_days", "st_nth_today", "st_log_elapsed", "st_log_i", "st_log_t"]
F_DECK = ["dk_acc_other", "dk_n_other", "dk_acc_all", "dk_n_all", "dk_cards_seen",
          "dk_since_other", "nt_acc_other", "nt_n_other"]
GROUPS = {"spine": F_SPINE, "surprise": F_SURPRISE, "state": F_STATE, "deck": F_DECK}
FEATURES = F_SPINE + F_SURPRISE + F_STATE + F_DECK
ID_COLS = ["user", "row_index", "fold", "y", "p_a", "p_b", "p_mixture", "p_fsrs",
           "card_id", "note_id", "deck_id", "day", "i_raw", "elapsed_days", "lapse"]
assert len(FEATURES) == len(set(FEATURES)), "duplicate feature name"

COMMANDS = {
    "build_lockbox": ".venv/bin/python evaluation/tabular-probe-20260917/probe.py "
                     "build --procs 4",
    "build_fresh": ".venv/bin/python evaluation/tabular-probe-20260917/probe.py "
                   "build --panel fresh --procs 3",
    "check": ".venv/bin/python evaluation/tabular-probe-20260917/probe.py check",
    "run": ".venv/bin/python evaluation/tabular-probe-20260917/probe.py run --dump-preds "
           "--procs 4",
    "spec": ".venv/bin/python evaluation/tabular-probe-20260917/probe.py spec [--panel fresh]",
    "external": ".venv/bin/python evaluation/tabular-probe-20260917/probe.py external "
                "--procs 4",
    "bench_device": ".venv/bin/python evaluation/tabular-probe-20260917/probe.py "
                    "bench-device --rounds 120 --procs 4",
    "wrapper": "systemd-run --user --unit=probe-<step> --collect -p MemoryMax=6G "
               "-p MemorySwapMax=0 -p Nice=5 -p CPUQuota=400% "
               "--working-directory=<repo> --setenv=OMP_NUM_THREADS=1 bash -lc '<cmd>'",
}


HARNESS = None


def _harness():
    """Repo config + feature engineer with the flags the full run used."""
    global HARNESS
    if HARNESS is None:
        sys.path.insert(0, str(ROOT / "srs-benchmark"))
        from sklearn.model_selection import TimeSeriesSplit
        from config import Config, create_parser
        from features.factory import create_feature_engineer
        import utils as bench_utils

        args, _ = create_parser().parse_known_args(
            ["--data", str(DATA), "--short", "--secs", "--recency",
             "--equalize_test_with_non_secs", "--algo", "FSRS-7"])
        HARNESS = (Config(args), create_feature_engineer, bench_utils, TimeSeriesSplit)
    return HARNESS


def scored_frame(user: int) -> pd.DataFrame:
    """The per-review rows the harness scored, in npz order, with card/day ids.

    Reproduces `features.create_features._create_features_with_equalized_test`:
    the scored stream is the secs frame restricted to the review_th of the
    non-secs frame's TimeSeriesSplit(5) test folds, concatenated fold-major --
    exactly how `save_prediction_cache` (run_full.py:93) writes each npz.
    """
    cfg, make_engineer, bench_utils, TimeSeriesSplit = _harness()
    rev = pd.read_parquet(DATA / "revlogs" / f"user_id={user}" / "data.parquet")

    # The two engineered frames (non-secs for the fold masks, secs for the scored
    # rows) are each several hundred MB for the largest users -- hold them one at a
    # time; the engineer copies its input, so `rev` can be reused.
    c = copy.deepcopy(cfg)
    c.use_secs_intervals = False
    non_secs = make_engineer(c).create_features(rev.copy())
    masks = [set(non_secs.iloc[te]["review_th"])
             for _, te in TimeSeriesSplit(n_splits=cfg.n_splits).split(non_secs)]
    del non_secs
    c = copy.deepcopy(cfg)
    c.use_secs_intervals = True
    secs = make_engineer(c).create_features(rev)
    del rev
    parts = []
    for fold, mask in enumerate(masks):
        sub = secs[secs["review_th"].isin(mask)].copy()
        sub["fold"] = np.uint8(fold)
        parts.append(sub)
    del secs
    frame = pd.concat(parts).reset_index(drop=True)
    del parts

    with np.load(NPZ_DIR / f"{user}.npz") as a:
        y, i_arr, days, lapse, fold = a["y"], a["i"], a["elapsed_days"], a["lapse"], a["fold"]
        p_fsrs = a["p"].astype(np.float64)
    assert len(frame) == len(y), f"user {user}: {len(frame)} rows vs npz {len(y)}"
    assert np.array_equal(frame["y"].to_numpy(np.int64), y.astype(np.int64)), user
    assert np.array_equal(frame["i"].to_numpy(np.int64), i_arr), user
    assert np.array_equal(frame["elapsed_days"].to_numpy(np.float64), days), user
    assert np.array_equal(frame["fold"].to_numpy(np.uint8), fold), user
    lapse_rt = np.array([bench_utils.count_lapse(r, t) for r, t in
                         zip(frame["r_history"], frame["t_history"])], dtype=np.int32)
    assert np.array_equal(lapse_rt, lapse), user
    frame["lapse"] = lapse_rt
    frame = frame.drop(columns=["r_history", "t_history", "t_history_secs",
                                "first_rating", "last_rating"], errors="ignore")

    cards = pd.read_parquet(DATA / "cards", filters=[("user_id", "=", user)],
                            columns=["card_id", "note_id", "deck_id"])
    frame = frame.merge(cards.drop_duplicates("card_id"), on="card_id", how="left")
    # ~5% of card ids in the revlogs have no row in data/cards (deleted cards).
    # Give each of them its own deck/note so no false sibling link is invented.
    missing = frame["deck_id"].isna().to_numpy()
    if missing.any():
        syn = -(frame["card_id"].to_numpy()[missing] + 1)
        frame.loc[missing, "deck_id"] = syn
        frame.loc[missing, "note_id"] = syn
    assert frame[["deck_id", "note_id"]].notna().all().all(), f"user {user}: unmapped card"
    assert (np.diff(fold) >= 0).all(), f"user {user}: folds not chronological"
    assert (np.diff(frame["day_offset"].to_numpy()) >= 0).all(), f"user {user}: days not monotone"
    frame["p_fsrs"] = p_fsrs
    return frame


# ---------------------------------------------------------------- feature math

def _prefix(x: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = _prefix(x)
    t = np.arange(len(x))
    return c[t] - c[np.maximum(0, t - w)]


def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, -np.inf)
    for j in range(1, min(w, n - 1) + 1):
        np.maximum(out[j:], x[:-j], out=out[j:])
    out[~np.isfinite(out)] = 0.0
    return out


def shrunk_base_windows(y: np.ndarray, p_fsrs: np.ndarray):
    """B's two features: shrunk rolling mean of the base residual y - p^F.

    `z_F + beta0 + beta1*x16 + beta2*x256` with z_F = logit(p^F) is B's logit, so
    this needs no p_B and is what makes the probe transportable to a panel whose
    B rows do not exist (external panel).
    """
    t = np.arange(len(y))
    rF = y - p_fsrs
    out = []
    for w in (16, 256):
        cw = np.minimum(t, w).astype(np.float64)
        out.append((_roll_sum(rF, w) / np.maximum(1.0, cw)) * cw / (cw + SHRINK))
    return out[0], out[1]


def crossfit_betas() -> dict[int, list[float]]:
    """Per-user B coefficients: `fits[g].coefficients` applies to `groups[g]`."""
    cf = json.loads(CROSSFIT.read_text())
    return {int(u): cf["fits"][g]["coefficients"]
            for g, grp in enumerate(cf["groups"]) for u in grp}


def derive_p_b(y, p_fsrs, beta):
    """logit p_B = logit(clip(p^F)) + beta0 + beta1*x16 + beta2*x256 (B's definition)."""
    x16, x256 = shrunk_base_windows(y, p_fsrs)
    z = np.log(np.clip(p_fsrs, CLIP, 1 - CLIP) / (1 - np.clip(p_fsrs, CLIP, 1 - CLIP)))
    return 1.0 / (1.0 + np.exp(-(z + beta[0] + beta[1] * x16 + beta[2] * x256)))


def compute_features(y, p_b, p_fsrs, i_arr, elapsed, lapse, nth_today, day,
                     card_code, deck_code, note_code) -> dict[str, np.ndarray]:
    """All features for one user.  Pure function of past rows only.

    Vectorized parts use exclusive prefix sums; the deck/note loop carries only
    running state of strictly earlier rows.
    """
    n = len(y)
    t = np.arange(n)
    # rF: base residual of the personalized FSRS-7 logit -- this is what B's own
    # x16/x256 are built from (the frozen feature memmaps in crossfit-equal-user,
    # verified against them below).  rB: residual of the published B prediction.
    rF = (y - p_fsrs).astype(np.float64)
    rB = (y - p_b).astype(np.float64)
    f: dict[str, np.ndarray] = {}

    # -- spine: shrunk rolling mean base residual (identical to run_full.design)
    f["x16"], f["x256"] = shrunk_base_windows(y, p_fsrs)

    # -- surprise: event-level residuals of the official FSRS-7 and of B
    for w in SURPRISE_WINDOWS:
        den = np.maximum(1.0, np.minimum(t, w).astype(np.float64))
        f[f"sF_mean{w}"] = _roll_sum(rF, w) / den
        f[f"sB_mean{w}"] = _roll_sum(rB, w) / den
        f[f"sF_max{w}"] = _roll_max(rF, w)
        f[f"sB_max{w}"] = _roll_max(rB, w)
    f["sF_last"] = np.concatenate(([0.0], rF[:-1])) if n else rF
    f["sB_last"] = np.concatenate(([0.0], rB[:-1])) if n else rB
    f["sF_all"] = _prefix(rF)[t] / np.maximum(1.0, t)
    f["sB_all"] = _prefix(rB)[t] / np.maximum(1.0, t)

    # -- state + deck/note: single causal pass
    keys = ("st_card_seq", "st_age_days", "dk_acc_other", "dk_n_other", "dk_acc_all",
            "dk_n_all", "dk_cards_seen", "dk_since_other", "nt_acc_other", "nt_n_other")
    out = {k: np.zeros(n, dtype=np.float64) for k in keys}
    d_cnt: dict = {}
    d_sum: dict = {}
    d_cards: dict = {}
    d_recent: dict = {}
    n_cnt: dict = {}
    n_sum: dict = {}
    c_cnt: dict = {}
    c_sum: dict = {}
    c_age: dict = {}
    for q in range(n):
        d, c, nt = deck_code[q], card_code[q], note_code[q]
        cnd = d_cnt.get(d, 0)
        smd = d_sum.get(d, 0.0)
        cnc = c_cnt.get(c, 0)
        smc = c_sum.get(c, 0.0)
        ntn = n_cnt.get(nt, 0)
        smn = n_sum.get(nt, 0.0)
        out["st_card_seq"][q] = cnc
        out["st_age_days"][q] = c_age.get(c, 0.0)
        out["dk_n_all"][q] = cnd
        out["dk_acc_all"][q] = smd / (cnd + SHRINK)
        no = cnd - cnc
        out["dk_n_other"][q] = no
        out["dk_acc_other"][q] = (smd - smc) / (no + SHRINK)
        out["dk_cards_seen"][q] = d_cards.get(d, 0)
        seen = d_recent.get(d)
        last_other = -1 if seen is None else (seen[1] if seen[0] != c else seen[3])
        out["dk_since_other"][q] = day[q] - day[last_other] if last_other >= 0 else -1.0
        nn = ntn - cnc
        out["nt_n_other"][q] = nn
        out["nt_acc_other"][q] = (smn - smc) / (nn + SHRINK)
        if seen is None:
            d_recent[d] = (c, q, -1, -1)
        else:
            d_recent[d] = (c, q, seen[2], seen[3]) if seen[0] == c else (c, q, seen[0], seen[1])
        if cnd == 0:
            d_cards[d] = 1
        elif cnc == 0:
            d_cards[d] = d_cards.get(d, 0) + 1
        d_cnt[d] = cnd + 1
        d_sum[d] = smd + rB[q]
        c_cnt[c] = cnc + 1
        c_sum[c] = smc + rB[q]
        c_age[c] = c_age.get(c, 0.0) + float(elapsed[q])
        n_cnt[nt] = ntn + 1
        n_sum[nt] = smn + rB[q]
    f.update(out)

    f["st_i"] = i_arr.astype(np.float64)
    f["st_elapsed_days"] = elapsed.astype(np.float64)
    f["st_lapse"] = lapse.astype(np.float64)
    f["st_t"] = t.astype(np.float64)
    f["st_nth_today"] = nth_today.astype(np.float64)
    f["st_log_elapsed"] = np.log1p(elapsed.astype(np.float64))
    f["st_log_i"] = np.log1p(i_arr.astype(np.float64))
    f["st_log_t"] = np.log1p(t.astype(np.float64))
    return {k: f[k].astype(np.float32) for k in FEATURES}


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codes(x: np.ndarray) -> np.ndarray:
    return pd.factorize(x, sort=False)[0].astype(np.int32)


def user_table(user: int, panel: str = "lockbox", betas: dict | None = None) -> pa.Table:
    """One user's scored rows + causal features.

    panel="lockbox": labels and `p_b` come from the aligned spine (and the derived
    `p_b` is cross-checked against it).  panel="fresh": the spine does not exist,
    so `p_b` is derived from the npz `p^F` and the published per-group crossfit
    coefficients (`crossfit_betas`); `p_a`/`p_mixture` are absent (NaN).
    """
    frame = scored_frame(user)
    y_np = frame["y"].to_numpy(np.float64)
    pf_np = frame["p_fsrs"].to_numpy(np.float64)
    args = (frame["i"].to_numpy(np.int64), frame["elapsed_days"].to_numpy(np.int64),
            frame["lapse"].to_numpy(np.int32), frame["nth_today"].to_numpy(np.int64),
            frame["day_offset"].to_numpy(np.int64), _codes(frame["card_id"].to_numpy()),
            _codes(frame["deck_id"].to_numpy()), _codes(frame["note_id"].to_numpy()))
    derived = derive_p_b(y_np, pf_np, betas[user])

    if panel == "lockbox":
        spine = pq.read_table(SPINE, filters=[("user", "=", user)]).to_pandas()
        spine = spine.sort_values("row_index", kind="stable")
        assert len(spine) == len(frame), f"user {user}: spine {len(spine)} vs frame {len(frame)}"
        for col in ("y", "i", "elapsed_days", "lapse", "fold"):
            assert np.array_equal(spine[col].to_numpy(), frame[col].to_numpy()), (user, col)
        p_b = spine["p_b"].to_numpy(np.float64)
        # transportability check: the derived B reproduces the published one
        assert np.abs(p_b - derived).max() < 1e-5, (user, np.abs(p_b - derived).max())
        p_a = spine["p_a"].to_numpy(np.float32)
        p_mixture = spine["p_mixture"].to_numpy(np.float32)
        row_index = spine["row_index"].to_numpy(np.int32)
    else:
        p_b = derived
        p_a = np.full(len(frame), np.nan, np.float32)
        p_mixture = np.full(len(frame), np.nan, np.float32)
        row_index = np.arange(len(frame), dtype=np.int32)

    feats = compute_features(y_np, p_b, pf_np, *args)
    cols = {
        "user": np.full(len(frame), user, np.int32),
        "row_index": row_index,
        "fold": frame["fold"].to_numpy(np.uint8),
        "y": frame["y"].to_numpy(np.int8),
        "p_a": p_a,
        "p_b": p_b.astype(np.float32),
        "p_mixture": p_mixture,
        "p_fsrs": frame["p_fsrs"].to_numpy(np.float32),
        "card_id": frame["card_id"].to_numpy(np.int64),
        "note_id": frame["note_id"].to_numpy(np.int64),
        "deck_id": frame["deck_id"].to_numpy(np.int64),
        "day": frame["day_offset"].to_numpy(np.int64),
        "i_raw": frame["i"].to_numpy(np.int32),
        "elapsed_days": frame["elapsed_days"].to_numpy(np.int32),
        "lapse": frame["lapse"].to_numpy(np.int32),
        **feats,
    }
    return pa.table(cols)


# ---------------------------------------------------------------- build

def _build_one(job):
    user, panel, betas = job
    return user, user_table(user, panel, betas)


def build(procs: int, users: list[int] | None = None, panel: str = "lockbox",
          out: Path = OUT_PARQUET) -> dict:
    if users is None:
        src = SPINE if panel == "lockbox" else FRESH_USERS
        users = json.loads(src.read_text()) if src.suffix == ".json" else \
            pq.read_table(src, columns=["user"]).column("user").to_numpy().tolist()
        if isinstance(users, dict):
            users = users.get("user_ids", users.get("users"))
    users = sorted({int(u) for u in users})
    betas = crossfit_betas()
    missing = [u for u in users if u not in betas]
    assert not missing, f"{len(missing)} users absent from crossfit.json groups: {missing[:5]}"
    t0 = time.time()
    schema = None
    writer = None
    done = 0
    tmp = out.with_suffix(".parquet.tmp")
    with ProcessPoolExecutor(max_workers=procs) as ex:
        jobs = [(u, panel, betas) for u in users]
        for user, table in ex.map(_build_one, jobs, chunksize=1):
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            writer.write_table(table)
            done += 1
            if done % 32 == 0 or done == len(users):
                print(f"  {done}/{len(users)} users  {time.time()-t0:.0f}s", flush=True)
    writer.close()
    tmp.replace(out)
    pf = pq.ParquetFile(out)
    info = {"panel": panel, "users": done, "rows": int(sum(
        pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))),
        "seconds": round(time.time() - t0, 1), "path": str(out)}
    return info


# ---------------------------------------------------------------- causality

def causality_check(users: list[int], rng_seed: int = 0, panel: str = "lockbox") -> dict:
    """Perturb rows >= t (labels and both model streams) -> row t must not move."""
    cfg, make_engineer, bench_utils, TimeSeriesSplit = _harness()
    rng = np.random.default_rng(rng_seed)
    res = {"users": [], "checks": 0, "max_abs_diff": 0.0}
    mm = {k: np.load(FEAT_MM / f"{k}.npy", mmap_mode="r") for k in ("r16", "r256")}
    chrono = json.loads((FEAT_MM / "chronology.json").read_text())
    order = [e["user"] for e in chrono["users"]]
    lens = np.array([e["rows"] for e in chrono["users"]], np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    idx = {u: k for k, u in enumerate(order)}

    for user in users:
        frame = scored_frame(user)
        y = frame["y"].to_numpy(np.float64)
        pf = frame["p_fsrs"].to_numpy(np.float64)
        if panel == "lockbox":
            spine = pq.read_table(SPINE, filters=[("user", "=", user)]).to_pandas()
            spine = spine.sort_values("row_index", kind="stable")
            pb = spine["p_b"].to_numpy(np.float64)
        else:
            pb = None
        args = (frame["i"].to_numpy(np.int64), frame["elapsed_days"].to_numpy(np.int64),
                frame["lapse"].to_numpy(np.int32), frame["nth_today"].to_numpy(np.int64),
                frame["day_offset"].to_numpy(np.int64),
                _codes(frame["card_id"].to_numpy()), _codes(frame["deck_id"].to_numpy()),
                _codes(frame["note_id"].to_numpy()))
        # transportable path: derive B from the crossfit coefficients and the npz
        # p^F alone, then recompute every feature from it (the fresh-panel path)
        beta = crossfit_betas()[user]
        derived = compute_features(y, derive_p_b(y, pf, beta), pf, *args)
        if pb is None:
            pb = derive_p_b(y, pf, beta)
        base = compute_features(y, pb, pf, *args)

        # spine features reproduce the frozen memmaps the crossfit consumed
        s, n = starts[idx[user]], lens[idx[user]]
        assert n == len(y), f"user {user}: memmap slice {n} vs frame {len(y)}"
        for key, name in (("r16", "x16"), ("r256", "x256")):
            t = np.arange(n)
            cw = np.minimum(t, 16 if key == "r16" else 256).astype(np.float64)
            ref = mm[key][s:s + n].astype(np.float64) * cw / (cw + SHRINK)
            assert np.abs(ref - base[name]).max() < 1e-5, (user, name)

        n_rows = len(y)
        for t in sorted({n_rows // 7, n_rows // 3, n_rows // 2, n_rows - 1}):
            if t < 1:
                continue
            y2, pb2, pf2 = y.copy(), pb.copy(), pf.copy()
            y2[t:] = 1.0 - y2[t:]
            pb2[t:] = rng.uniform(0.05, 0.95, size=n_rows - t)
            pf2[t:] = rng.uniform(0.05, 0.95, size=n_rows - t)
            perturbed = compute_features(y2, pb2, pf2, *args)
            derived_p = compute_features(y2, derive_p_b(y2, pf2, beta), pf2, *args)
            diffs = {k: float(np.abs(perturbed[k][t:t + 1] - base[k][t:t + 1]).max())
                     for k in FEATURES}
            diffs.update({f"derived:{k}": float(np.abs(
                derived_p[k][t:t + 1] - derived[k][t:t + 1]).max()) for k in FEATURES})
            worst = max(diffs.values())
            res["checks"] += 1
            res["max_abs_diff"] = max(res["max_abs_diff"], worst)
            assert worst == 0.0, (user, t, {k: v for k, v in diffs.items() if v > 0})
        res["users"].append({"user": user, "rows": n_rows})
    return res


# ---------------------------------------------------------------- probe

def bin_grouping(elapsed_days, i_arr, lapse):
    """Copied from evaluation/full-benchmark-9999-20260916/run_full.py:397 (rmse_matrix)."""
    dt = np.round(2.48 * np.power(3.62, np.floor(
        np.log(np.maximum(elapsed_days, 1e-6)) / np.log(3.62))), 2)
    ii = np.round(1.99 * np.power(1.89, np.floor(
        np.log(i_arr.astype(np.float64)) / np.log(1.89))), 0)
    ll = np.where(lapse == 0, 0., np.round(1.65 * np.power(1.73, np.floor(
        np.log(np.maximum(lapse, 1).astype(np.float64)) / np.log(1.73))), 0))
    _, inv = np.unique(np.stack([dt, ii, ll], 1), axis=0, return_inverse=True)
    return inv, np.bincount(inv).astype(np.float64)


def binned_rmse(inv, cnt, y, p):
    ym = np.bincount(inv, weights=y) / cnt
    pm = np.bincount(inv, weights=p) / cnt
    return float(np.sqrt(np.average((ym - pm) ** 2, weights=cnt)))


def per_user_metrics(y, p, grouping, size):
    inv, cnt = grouping
    try:
        # roc_auc_score returns nan (not raises) for a single-class user, as in the
        # fresh panel's 3 degenerate users; the corpus aggregation drops those too.
        auc = float(roc_auc_score(y, p))
        if not np.isfinite(auc):
            auc = None
    except Exception:                    # single-class user
        auc = None
    # LogLoss goes in unclipped: this is the benchmark's own `log_loss(y, p, labels=[0, 1])`
    # (see full-benchmark-9999-20260916/run_full.py), which clips at 1e-15 internally.  CLIP
    # (1e-6) is a *logit* guard for base margins -- applying it here would report a smaller
    # loss than the benchmark for confident misses (audit 2026-09-19, section 3).
    return {"LogLoss": float(log_loss(y, p, labels=[0, 1])),
            "RMSE(bins)": binned_rmse(inv, cnt, y, p),
            "AUC": auc, "MBE": float(np.mean(p - y)), "size": int(size)}


def equal_user(per_user, metric):
    vals = [r[metric] for r in per_user if r[metric] is not None and np.isfinite(r[metric])]
    return float(np.mean(vals)) if vals else float("nan")


def paired_bootstrap(a, b, n_boot=20000, seed=0):
    """Per-user LL deltas a-b; users are the resampling unit."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    k = len(d)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = d[idx].mean(axis=1)
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "se": float(boots.std(ddof=1)),
            "improved": int((d < 0).sum()), "worsened": int((d > 0).sum()),
            "tied": int((d == 0).sum()), "n": int(k)}


def offset_logreg(X, y, z, w, l2=1e-3, maxiter=400):
    """Equal-user weighted logistic regression with a fixed offset z (base margin)."""
    from scipy.optimize import minimize

    def obj(theta):
        eta = z + X @ theta
        val = float(np.dot(w, np.logaddexp(0.0, eta) - y * eta) + 0.5 * l2 * theta @ theta)
        g = X.T @ (w * (1.0 / (1.0 + np.exp(-eta)) - y)) + l2 * theta
        return val, g

    r = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter})
    return r.x


ID_DTYPES = {"user": np.int32, "row_index": np.int32, "fold": np.uint8, "y": np.int8,
             "p_a": np.float32, "p_b": np.float32, "p_mixture": np.float32,
             "p_fsrs": np.float32, "card_id": np.int64, "note_id": np.int64,
             "deck_id": np.int64, "day": np.int64, "i_raw": np.int32,
             "elapsed_days": np.int32, "lapse": np.int32}


def load_matrix(path: Path, feats: list[str], id_cols: list[str], chunk: int = 500_000):
    """Stream a features parquet into one float32 design matrix + id vectors.

    Row-batch streaming keeps peak memory at the matrix itself (the earlier
    `read_table().to_numpy()` + `column_stack` route doubled it and OOM'd a 6 GB
    cgroup cap).
    """
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    X = np.empty((n, len(feats)), np.float32)
    ids = {c: np.empty(n, ID_DTYPES[c]) for c in id_cols}
    row = 0
    for batch in pf.iter_batches(batch_size=chunk, columns=list(id_cols) + list(feats)):
        m = batch.num_rows
        for j, name in enumerate(feats):
            X[row:row + m, j] = batch.column(name).to_numpy()
        for c in id_cols:
            ids[c][row:row + m] = batch.column(c).to_numpy()
        row += m
    assert row == n, (row, n)
    return X, ids


def split_users(sizes: dict[int, int]) -> dict:
    """Volume-interleaved user split: 128 fit / 128 eval, no user on both sides."""
    order = sorted(sizes, key=lambda u: (sizes[u], u))
    fit, ev = order[0::2], order[1::2]
    val = fit[3::4]
    keep = set(val)
    return {"train": [u for u in fit if u not in keep], "val": val, "eval": ev}


def equal_user_weights(user_of: np.ndarray, sizes: dict[int, int]) -> np.ndarray:
    """1/n_u (equal-user) weights scaled to mean 1.

    XGBoost's `min_child_weight` and `reg_lambda` are sums of *weighted* hessians,
    so against raw 1/n_u (~2.7e-5 here) a lambda of 1 swamps every leaf
    (leaf = -G/(H+lambda) with G, H ~ 2.7e-5 * sum): the booster never leaves the
    base margin -- `best_iteration=0` and every variant identical to B.  Scaling by
    1/mean keeps the weights proportional (the equal-user objective is untouched)
    and restores the documented units: min_child_weight=20 ~ 20 average rows,
    reg_lambda=1 ~ 1 average row.
    """
    w = np.array([1.0 / sizes[int(u)] for u in user_of], np.float32)
    w /= w.mean()
    return w


VARIANT_GROUPS = [("B", []), ("spine", F_SPINE), ("surprise", F_SURPRISE),
                  ("state", F_STATE), ("deck", F_DECK),
                  ("spine+surprise", F_SPINE + F_SURPRISE),
                  ("spine+surprise+state", F_SPINE + F_SURPRISE + F_STATE),
                  ("all", FEATURES)]


def run(device: str, rounds: int, depth: int, eta: float, procs: int, seed: int,
        dump_preds: bool = False) -> dict:
    import xgboost as xgb

    id_cols = ["user", "row_index", "fold", "y", "p_a", "p_b", "p_mixture", "p_fsrs",
               "elapsed_days", "i_raw", "lapse"]
    Xall, d = load_matrix(OUT_PARQUET, FEATURES, id_cols)
    n_rows = len(d["y"])
    user_of = d["user"].astype(np.int64)
    sizes = {int(k): int(v) for k, v in zip(*np.unique(user_of, return_counts=True))}
    split = split_users(sizes)
    masks = {k: np.isin(user_of, np.asarray(v, np.int64)) for k, v in split.items()}
    assert not (masks["train"] & masks["eval"]).any()
    eval_preds: dict[str, np.ndarray] = {}

    y_all = d["y"].astype(np.float64)
    p_b = d["p_b"].astype(np.float64)
    pb = np.clip(p_b, CLIP, 1 - CLIP)
    z_b = np.log(pb / (1 - pb)).astype(np.float32)
    w_all = equal_user_weights(user_of, sizes)

    ev_order = sorted(split["eval"])
    ev_rows = {int(u): np.flatnonzero(masks["eval"] & (user_of == u)) for u in ev_order}
    grouping = {}
    for u in ev_order:
        r = ev_rows[u]
        grouping[u] = bin_grouping(np.maximum(d["elapsed_days"][r].astype(np.float64), 1e-6),
                                   np.maximum(d["i_raw"][r].astype(np.int64), 1),
                                   d["lapse"][r].astype(np.int64))

    def score(p_full):
        return [per_user_metrics(y_all[r], p_full[r], grouping[u], len(r))
                for u, r in ((u, ev_rows[u]) for u in ev_order)]

    def summarize(p_full):
        per = score(p_full)
        return {"metrics": {m: equal_user(per, m) for m in
                            ("LogLoss", "RMSE(bins)", "AUC", "MBE")},
                "per_user": [{k: (None if v is None else float(v)) for k, v in r.items()}
                             for r in per],
                "per_user_LogLoss": [float(r["LogLoss"]) for r in per]}

    results = {
        "panel": str(SPINE), "features_all": str(OUT_PARQUET), "rows": int(n_rows),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "split_rows": {k: int(m.sum()) for k, m in masks.items()},
        "split_users": {k: [int(u) for u in v] for k, v in split.items()},
        "device": device, "rows_clipped_for_offset": int(((p_b <= CLIP) | (p_b >= 1 - CLIP)).sum()),
        "xgb": {"num_boost_round": rounds, "max_depth": depth, "eta": eta,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20.0,
                "reg_lambda": 1.0, "early_stopping_rounds": 30, "seed": seed,
                "row_weight": "1/n_u scaled to mean 1 (equal-user; raw 1/n_u makes "
                              "min_child_weight/reg_lambda swamp every leaf)", "base_margin": "logit(clip(p_b))"},
        "variants": {}, "references": {}, "commands": COMMANDS,
    }

    # references on the same eval users -------------------------------------
    for name, col in (("B", "p_b"), ("A", "p_a"), ("FSRS-7", "p_fsrs"),
                      ("published_mixture", "p_mixture")):
        p_full = np.zeros(n_rows)
        p_full[masks["eval"]] = d[col][masks["eval"]].astype(np.float64)
        results["references"][name] = summarize(p_full)["metrics"]

    # XGBoost ablation -------------------------------------------------------
    params = {"objective": "binary:logistic", "tree_method": "hist", "device": device,
              "max_depth": depth, "eta": eta, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 20.0, "reg_lambda": 1.0,
              "eval_metric": "logloss", "seed": seed, "nthread": procs}
    for name, feats in VARIANT_GROUPS:
        t0 = time.time()
        model = None
        if not feats:
            p_full = np.zeros(n_rows)
            p_full[masks["eval"]] = p_b[masks["eval"]]
            best = None
        else:
            idx = [FEATURES.index(f) for f in feats]
            Xtr = np.ascontiguousarray(Xall[np.ix_(masks["train"], idx)])
            Xva = np.ascontiguousarray(Xall[np.ix_(masks["val"], idx)])
            dtr = xgb.QuantileDMatrix(
                Xtr, label=d["y"][masks["train"]].astype(np.float32),
                weight=w_all[masks["train"]], base_margin=z_b[masks["train"]],
                feature_names=list(feats), nthread=procs, missing=np.nan)
            dva = xgb.QuantileDMatrix(
                Xva, label=d["y"][masks["val"]].astype(np.float32),
                weight=w_all[masks["val"]], base_margin=z_b[masks["val"]],
                ref=dtr, feature_names=list(feats), nthread=procs, missing=np.nan)
            bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "val")],
                            early_stopping_rounds=30, verbose_eval=False)
            best = int(getattr(bst, "best_iteration", rounds - 1))
            del dtr, dva, Xtr, Xva
            Xev = np.ascontiguousarray(Xall[np.ix_(masks["eval"], idx)])
            dev = xgb.DMatrix(Xev, base_margin=z_b[masks["eval"]],
                              feature_names=list(feats), nthread=procs, missing=np.nan)
            pred = bst.predict(dev, iteration_range=(0, best + 1))
            p_full = np.zeros(n_rows)
            p_full[masks["eval"]] = pred
            del dev, Xev
            bst.set_attr(best_iteration=str(best))
            model = HERE / f"model-{name.replace('+', '_')}.json"
            bst.save_model(model)
        out = summarize(p_full)
        out.update({"features": feats, "seconds": round(time.time() - t0, 1),
                    "best_iteration": best, "rows_at_clip": int(
                        ((p_full <= CLIP) | (p_full >= 1 - CLIP)).sum()),
                    "model": (None if model is None else str(model)),
                    "model_sha256": (None if model is None else sha256(model))})
        results["variants"][name] = out
        eval_preds[name] = p_full[masks["eval"]].astype(np.float32)
        print(f"  {name:22s} {out['metrics']['LogLoss']:.6f}  {out['seconds']}s", flush=True)

    # logistic-regression reference -----------------------------------------
    t0 = time.time()
    tr, ev = masks["train"], masks["eval"]
    allc = range(len(FEATURES))
    # standardized in place: the naive (X - mu) / sd allocates two full float64
    # copies of a 3-5 M row matrix and was the largest transient in the run
    Xtr = Xall[np.ix_(tr, allc)].astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr -= mu
    Xtr /= sd
    th = offset_logreg(Xtr, y_all[tr], z_b[tr].astype(np.float64),
                       w_all[tr].astype(np.float64))
    del Xtr
    Eev = Xall[np.ix_(ev, allc)].astype(np.float64)
    Eev -= mu
    Eev /= sd
    p_lr = np.zeros(n_rows)
    p_lr[ev] = 1.0 / (1.0 + np.exp(-(z_b[ev].astype(np.float64) + Eev @ th)))
    del Eev
    out = summarize(p_lr)
    out.update({"features": FEATURES, "seconds": round(time.time() - t0, 1),
                "coefficients_standardized": th.tolist()})
    results["variants"]["lr_all"] = out
    eval_preds["lr_all"] = p_lr[masks["eval"]].astype(np.float32)
    print(f"  {'lr_all':22s} {out['metrics']['LogLoss']:.6f}  {out['seconds']}s", flush=True)

    # per-row predictions on the eval users (spine order: user, then row_index)
    # plus the lr reference -- what a downstream stacker needs to test the probe
    # on top of an existing combiner without re-deriving any feature.
    if dump_preds:
        ev_r = np.flatnonzero(masks["eval"])
        cols_out = {
            "user": d["user"][ev_r].astype(np.int32),
            "row_index": d["row_index"][ev_r].astype(np.int32),
            "fold": d["fold"][ev_r].astype(np.uint8),
            "y": d["y"][ev_r].astype(np.int8),
            "p_b": d["p_b"][ev_r].astype(np.float32),
            "p_a": d["p_a"][ev_r].astype(np.float32),
            "p_fsrs": d["p_fsrs"][ev_r].astype(np.float32),
            "p_mixture": d["p_mixture"][ev_r].astype(np.float32),
        }
        for vname, pred in eval_preds.items():
            cols_out[f"p_{vname.replace('+', '_')}"] = pred
        order = np.lexsort((d["row_index"][ev_r], d["user"][ev_r]))
        pq.write_table(pa.table({k: v[order] for k, v in cols_out.items()}),
                       HERE / "preds-eval.parquet", compression="zstd")
        results["preds_eval"] = {"path": str(HERE / "preds-eval.parquet"),
                                 "rows": int(len(ev_r)),
                                 "columns": list(cols_out),
                                 "note": "eval users only, one row per scored review in "
                                         "spine order (user, row_index); the model outputs "
                                         "already include the B offset as base_margin"}

    # paired deltas ----------------------------------------------------------
    b_ll = np.array(results["variants"]["B"]["per_user_LogLoss"])
    results["paired_vs_B"] = {
        name: paired_bootstrap(np.array(v["per_user_LogLoss"]), b_ll)
        for name, v in results["variants"].items() if name != "B"}
    inc = {}
    seq = ["spine", "spine+surprise", "spine+surprise+state", "all"]
    for prev, cur in zip(["B"] + seq[:-1], seq):
        inc[f"{cur} - {prev}"] = paired_bootstrap(
            np.array(results["variants"][cur]["per_user_LogLoss"]),
            np.array(results["variants"][prev]["per_user_LogLoss"]))
    for name in ("surprise", "state", "deck", "lr_all"):
        inc[f"{name} - B"] = paired_bootstrap(
            np.array(results["variants"][name]["per_user_LogLoss"]), b_ll)
    results["incremental"] = inc

    # official promotion gate (evaluation/promotion-baseline-20260917/gate_report.py)
    # on the same 128 eval users: paired CI vs the published B / FSRS-7 / mixture(A,B).
    import subprocess
    gate_cmd = ["--panel", str(PANEL_JSON), "--label"]
    results["gate"] = {}
    for name, v in results["variants"].items():
        if name == "B":
            continue
        safe = name.replace("+", "_")
        cand = HERE / f"result-{safe}.jsonl"
        with cand.open("w") as fh:
            for u, r in zip(ev_order, v["per_user"]):
                fh.write(json.dumps({"user": int(u), "size": int(r["size"]),
                                     "metrics": r}) + "\n")
        out = HERE / f"gate-{safe}.json"
        cmd = [sys.executable, str(GATE), "--candidate", str(cand), "--out", str(out),
               *gate_cmd, f"tabular-probe:{safe}"]
        subprocess.run(cmd, check=True, capture_output=True)
        g = json.loads(out.read_text())
        results["gate"][name] = {
            "shape": f"official gate_report.py on {g['users']} users / {g['rows']} rows",
            "verdict": g["verdict"], "deltas": g["deltas"],
            "equal_user": g["equal_user"]}
    (HERE / "results.json").write_text(json.dumps(results, indent=1, allow_nan=False))
    return results


def provenance(procs: int, panel: str = "lockbox", path: Path | None = None) -> dict:
    """Machine-readable feature spec + build provenance for reuse by other tools.

    Writes feature-spec.json (columns, dtypes, groups, clip/base-margin convention)
    and build-info.json (row/user counts, frame-source check, id coverage).
    """
    parquet = path or (OUT_PARQUET if panel == "lockbox" else FRESH_PARQUET)
    pf = pq.ParquetFile(parquet)
    schema = pf.schema_arrow
    table = pq.read_table(parquet, columns=["user", "row_index", "y", "fold",
                                            "card_id", "deck_id", "day"])
    users = table.column("user").to_numpy()
    n_rows = len(users)
    n_users = int(len(np.unique(users)))

    # per-user row counts: this parquet vs the spine (lockbox) and the npz frames
    own_counts = pd.Series(users).value_counts().to_dict()
    spine_counts = {}
    if panel == "lockbox":
        spine = pq.read_table(SPINE, columns=["user", "row_index"]).to_pandas()
        spine_counts = spine.groupby("user").size().to_dict()
        assert np.array_equal(
            spine.sort_values(["user", "row_index"])["user"].to_numpy(), np.sort(users)), \
            "user/row_index pairing differs from the spine"
    npz_counts = {}
    for u in sorted(own_counts):
        with np.load(NPZ_DIR / f"{u}.npz") as a:
            npz_counts[int(u)] = int(a["y"].shape[0])
    mism = {int(u): [int(own_counts[u]), int(spine_counts.get(u, -1)), npz_counts[int(u)]]
            for u in own_counts
            if int(own_counts[u]) != npz_counts[int(u)]
            or (panel == "lockbox" and int(own_counts[u]) != int(spine_counts.get(u, -1)))}
    # row_index must be 0..n-1 within each user, in order (stream order preserved)
    ri = table.column("row_index").to_numpy()
    contiguous = bool(np.all(ri == np.concatenate(
        [np.arange(c) for c in pd.Series(users).value_counts().sort_index().to_numpy()])))
    if panel == "fresh":
        ref_rows = json.loads(FRESH_REF.read_text())["rows"]
        assert n_rows == ref_rows, (n_rows, ref_rows)
    unmapped_rows = int((table.column("deck_id").to_numpy() < 0).sum())
    unmapped_cards = int(len(np.unique(table.column("card_id").to_numpy()
                                       [table.column("deck_id").to_numpy() < 0])))

    feat_spec = {
        "panel_name": panel,
        "parquet": str(parquet),
        "parquet_bytes": parquet.stat().st_size,
        "rows": n_rows, "users": n_users, "row_groups": pf.metadata.num_row_groups,
        "built_by": f"{Path(__file__).name} build --panel {panel} --procs {procs}",
        "panel": str(SPINE if panel == "lockbox" else FRESH_DIR),
        "panel_json": str(PANEL_JSON if panel == "lockbox" else FRESH_USERS),
        "p_b_source": ("published spine p_b (aligned-panel2.parquet), cross-checked "
                       "against the derived one" if panel == "lockbox" else
                       "derived: expit(logit(clip(p^F)) + b0 + b1*x16 + b2*x256) with "
                       "per-group coefficients from crossfit.json"),
        "frame_source": str(NPZ_DIR / "<user>.npz"),
        "row_order": "spine order: (user ascending, row_index ascending); row_index is "
                     "the npz stream position (fold-major concat of the 5 equalized "
                     "test folds), verified against the npz per user",
        "id_columns": {c: str(schema.field(c).type) for c in ID_COLS},
        "feature_groups": {g: v for g, v in GROUPS.items()},
        "feature_dtypes": {c: str(schema.field(c).type) for c in FEATURES},
        "features_flat": FEATURES,
        "offset": {"column": "p_b", "base_margin": "logit(clip(p_b, CLIP, 1-CLIP))",
                   "CLIP": CLIP,
                   "note": "the booster predicts the CORRECTION on top of this offset; "
                           "final p = expit(logit(clip(p_b)) + booster_output)"},
        "causality": {"statement": "every feature of row t uses rows < t only",
                      "check": "probe.py check (perturb rows >= t, features for t "
                               "bit-identical), see check-causality.json"},
        "shrinkage_k": SHRINK, "surprise_windows": list(SURPRISE_WINDOWS),
        "deck_note_ids": {
            "unmapped_rows": unmapped_rows, "unmapped_cards": unmapped_cards,
            "unmapped_share": round(unmapped_rows / n_rows, 6),
            "convention": "cards absent from data/cards get singleton negative "
                          "deck_id/note_id (no invented sibling links)"},
        "split": ({"unit": "user", "fit_users": 128, "eval_users": 128,
                   "definition": "users ranked by row count; even rank = fit "
                                 "(every 4th of those = early-stopping validation), "
                                 "odd rank = eval"} if panel == "lockbox" else
                  {"unit": "user", "eval_users": n_users,
                   "definition": "holdout panel: every user is eval; nothing is "
                                 "fitted here, the boosters come from the lockbox "
                                 "96 train / 32 early-stopping split"}),
        "models": {},
    }
    build_info = {
        "built_by": feat_spec["built_by"], "rows": n_rows, "users": n_users,
        "per_user_row_count_mismatches_vs_spine_and_npz": mism,
        "row_index_0based_contiguous_per_user": contiguous,
        "checks": {
            "spine_row_counts_equal": len(mism) == 0,
            "labels_equal_to_spine_and_npz": True,
            "i_elapsed_days_lapse_fold_equal_to_npz": True,
            "fold_non_decreasing_per_user": True,
            "day_offset_non_decreasing_per_user": True,
            "x16_x256_equal_to_frozen_memmaps": "<=1e-5 (float32 memmap rounding)",
        },
        "unmapped_card_rows": unmapped_rows, "unmapped_cards": unmapped_cards,
        "note": "the three *_equal checks are asserted inside user_table() for every "
                "user during build; mismatch maps here were recomputed from the "
                "written parquet, the spine and the npz caches",
    }
    for fname in sorted(HERE.glob("model-*.json")):
        feat_spec["models"][fname.stem.replace("model-", "")] = {
            "path": str(fname), "sha256": sha256(fname)}
    suffix = "" if panel == "lockbox" else "-fresh"
    fs_path, bi_path = HERE / f"feature-spec{suffix}.json", HERE / f"build-info{suffix}.json"
    fs_path.write_text(json.dumps(feat_spec, indent=1))
    bi_path.write_text(json.dumps(build_info, indent=1))
    return {"feature_spec": str(fs_path), "build_info": str(bi_path),
            "rows": n_rows, "users": n_users,
            "row_count_mismatches": len(mism), "unmapped_card_rows": unmapped_rows}


def external(models: list[str] | None, device: str, procs: int,
             features: Path = FRESH_PARQUET, refs_path: Path = FRESH_REF,
             preds: Path = HERE / "preds-fresh.parquet",
             out: Path = HERE / "results-fresh.json") -> dict:
    """Apply the lockbox-trained boosters to an untouched panel (external test).

    Nothing is fitted here: the boosters are loaded from `model-*.json`, the only
    panel-specific input is `p_b`, derived from the npz `p^F` plus the published
    per-group crossfit coefficients.  Pairing is per user against the published
    B / FSRS-7 per-user metrics in `references.json`.
    """
    import subprocess
    import xgboost as xgb

    id_cols = ["user", "row_index", "fold", "y", "p_b", "p_fsrs", "elapsed_days",
               "i_raw", "lapse"]
    Xall, d = load_matrix(features, FEATURES, id_cols)
    n_rows = len(d["y"])
    user_of = d["user"].astype(np.int64)
    y_all = d["y"].astype(np.float64)
    pb = np.clip(d["p_b"].astype(np.float64), CLIP, 1 - CLIP)
    z_b = np.log(pb / (1 - pb)).astype(np.float32)
    users = sorted(set(int(u) for u in user_of))
    rows = {u: np.flatnonzero(user_of == u) for u in users}
    grouping = {u: bin_grouping(np.maximum(d["elapsed_days"][r].astype(np.float64), 1e-6),
                                np.maximum(d["i_raw"][r].astype(np.int64), 1),
                                d["lapse"][r].astype(np.int64)) for u, r in rows.items()}

    def summarize(p):
        per = [per_user_metrics(y_all[r], p[r], grouping[u], len(r))
               for u, r in rows.items()]
        return {"metrics": {m: equal_user(per, m) for m in
                            ("LogLoss", "RMSE(bins)", "AUC", "MBE")},
                "per_user": [{k: (None if v is None else float(v)) for k, v in q.items()}
                             for q in per],
                "per_user_LogLoss": [float(q["LogLoss"]) for q in per]}, per

    refs = json.loads(refs_path.read_text())["per_user_metrics"]
    results = {"features": str(features), "rows": int(n_rows), "users": len(users),
               "device": device, "models": models, "references": {}, "verification": {},
               "variants": {}, "paired_vs_B": {}, "commands": COMMANDS}

    # our own reproduction of the published B and FSRS-7 on this panel
    b_sum, _ = summarize(pb)
    f_sum, _ = summarize(np.clip(d["p_fsrs"].astype(np.float64), CLIP, 1 - CLIP))
    for name, mine, key in (("B", b_sum, "b_crossfit"), ("FSRS-7", f_sum, "fsrs7")):
        ref = refs[key]
        diffs = [abs(mine["per_user_LogLoss"][i] - float(ref[str(u)]["LogLoss"]))
                 for i, u in enumerate(users) if str(u) in ref]
        results["references"][name] = {
            "mine": mine["metrics"],
            "published": {m: float(np.mean([float(ref[str(u)][m]) for u in users
                                            if str(u) in ref and ref[str(u)][m] is not None]))
                          for m in ("LogLoss", "RMSE(bins)", "AUC", "MBE")},
            "per_user_LogLoss_max_abs_diff": float(max(diffs)) if diffs else None,
            "users_compared": len(diffs)}
    results["verification"]["my_B_matches_published"] = bool(
        results["references"]["B"]["per_user_LogLoss_max_abs_diff"] < 1e-4)

    # published A and mixture are not available per row on this panel; their
    # per-user files do cover these users, so the official gate can still use them.
    models = models or [f.stem.replace("model-", "") for f in
                        sorted(HERE.glob("model-*.json"))]
    pred_cols = {"user": d["user"].astype(np.int32), "row_index": d["row_index"].astype(np.int32),
                 "fold": d["fold"].astype(np.uint8), "y": d["y"].astype(np.int8),
                 "p_b": d["p_b"].astype(np.float32),
                 "p_fsrs": d["p_fsrs"].astype(np.float32)}
    for name in models:
        path = HERE / f"model-{name}.json"
        bst = xgb.Booster()
        bst.load_model(path)
        feats = list(bst.feature_names)
        assert feats and all(f in FEATURES for f in feats), (name, feats)
        X = np.ascontiguousarray(Xall[:, [FEATURES.index(f) for f in feats]])
        dm = xgb.DMatrix(X, base_margin=z_b, feature_names=feats, nthread=procs,
                         missing=np.nan)
        best = bst.attr("best_iteration")
        it = (0, int(best) + 1) if best is not None else (0, 0)
        p = bst.predict(dm, iteration_range=it)
        del X, dm
        summ, _ = summarize(np.clip(p.astype(np.float64), CLIP, 1 - CLIP))
        summ.update({"features": feats, "model": str(path), "sha256": sha256(path),
                     "best_iteration": (int(best) if best is not None else None)})
        results["variants"][name] = summ
        pred_cols[f"p_{name}"] = p.astype(np.float32)
        print(f"  {name:22s} {summ['metrics']['LogLoss']:.6f}", flush=True)

    b_ref_ll = np.array([float(refs["b_crossfit"][str(u)]["LogLoss"]) for u in users])
    b_own_ll = np.array(b_sum["per_user_LogLoss"])
    for name, v in results["variants"].items():
        results["paired_vs_B"][name] = {
            "vs_published_B": paired_bootstrap(np.array(v["per_user_LogLoss"]), b_ref_ll),
            "vs_our_B": paired_bootstrap(np.array(v["per_user_LogLoss"]), b_own_ll)}
    order = np.lexsort((pred_cols["row_index"], pred_cols["user"]))
    pq.write_table(pa.table({k: v[order] for k, v in pred_cols.items()}), preds,
                   compression="zstd")
    results["preds"] = {"path": str(preds), "rows": int(n_rows), "columns": list(pred_cols)}

    # official gate for completeness (its A / B / FSRS-7 refs cover these users)
    panel_file = HERE / "fresh-panel-users.json"
    panel_file.write_text(json.dumps({"user_ids": [int(u) for u in users]}))
    results["gate"] = {}
    for name, v in results["variants"].items():
        cand = HERE / f"result-fresh-{name}.jsonl"
        with cand.open("w") as fh:
            for u, q in zip(users, v["per_user"]):
                fh.write(json.dumps({"user": int(u), "size": int(q["size"]),
                                     "metrics": q}) + "\n")
        gout = HERE / f"gate-fresh-{name}.json"
        subprocess.run([sys.executable, str(GATE), "--candidate", str(cand),
                        "--panel", str(panel_file), "--label", f"tabular-probe-fresh:{name}",
                        "--out", str(gout)], check=True, capture_output=True)
        g = json.loads(gout.read_text())
        results["gate"][name] = {"verdict": g["verdict"], "deltas": g["deltas"],
                                 "equal_user": g["equal_user"]}
    out.write_text(json.dumps(results, indent=1, allow_nan=False))
    return results


def bench_device(rounds: int, procs: int, seed: int) -> dict:
    """Time the same XGBoost fit on cpu vs cuda (matrix build + training)."""
    import xgboost as xgb

    feats = F_SPINE + F_SURPRISE + F_STATE
    Xall, d = load_matrix(OUT_PARQUET, feats, ["user", "y", "p_b"])
    user_of = d["user"].astype(np.int64)
    sizes = {int(k): int(v) for k, v in zip(*np.unique(user_of, return_counts=True))}
    split = split_users(sizes)
    masks = {k: np.isin(user_of, np.asarray(v, np.int64)) for k, v in split.items()}
    p_b = np.clip(d["p_b"].astype(np.float64), CLIP, 1 - CLIP)
    z_b = np.log(p_b / (1 - p_b)).astype(np.float32)
    w_all = equal_user_weights(user_of, sizes)
    Xtr = np.ascontiguousarray(Xall[masks["train"]])
    del Xall
    out = {"features": feats, "rows": int(masks["train"].sum()), "rounds": rounds,
           "procs": procs, "timings": {}}
    for device in ("cpu", "cuda"):
        params = {"objective": "binary:logistic", "tree_method": "hist", "device": device,
                  "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 20.0, "reg_lambda": 1.0, "seed": seed,
                  "nthread": procs}
        try:
            t0 = time.time()
            dtr = xgb.QuantileDMatrix(Xtr, label=d["y"][masks["train"]].astype(np.float32),
                                      weight=w_all[masks["train"]],
                                      base_margin=z_b[masks["train"]], nthread=procs)
            t_matrix = time.time() - t0
            t0 = time.time()
            xgb.train(params, dtr, num_boost_round=rounds, verbose_eval=False)
            t_fit = time.time() - t0
            out["timings"][device] = {"quantile_dmatrix_s": round(t_matrix, 2),
                                      "fit_s": round(t_fit, 2),
                                      "total_s": round(t_matrix + t_fit, 2),
                                      "per_round_s": round(t_fit / rounds, 4)}
            del dtr
        except Exception as exc:                       # noqa: BLE001
            out["timings"][device] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    (HERE / "bench-device.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "run", "check", "bench-device", "spec",
                                   "external"])
    ap.add_argument("--panel", default="lockbox", choices=["lockbox", "fresh"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--users", type=int, nargs="*", default=None)
    ap.add_argument("--causality-users", type=int, nargs="*", default=None,
                    help="default: 5 lockbox users, or 5 spread fresh-panel users")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dump-preds", action="store_true",
                    help="also write preds-eval.parquet (per-row eval predictions)")
    args = ap.parse_args()
    if args.causality_users is None:
        if args.panel == "lockbox":
            args.causality_users = [16, 21, 32, 61, 72]
        else:
            fresh = sorted(int(u) for u in json.loads(FRESH_USERS.read_text())["users"])
            args.causality_users = fresh[::max(1, len(fresh) // 5)][:5]
    if args.cmd == "build":
        out = args.out or (OUT_PARQUET if args.panel == "lockbox" else FRESH_PARQUET)
        print(json.dumps(build(args.procs, args.users, args.panel, out), indent=1))
        print(json.dumps(causality_check(args.causality_users, panel=args.panel), indent=1))
    elif args.cmd == "external":
        ext = external(args.models, args.device, args.procs)
        print(json.dumps({k: ext[k] for k in ("rows", "users", "references", "gate")},
                         indent=1)[:4000])
    elif args.cmd == "spec":
        print(json.dumps(provenance(args.procs, args.panel, args.out), indent=1))
    elif args.cmd == "bench-device":
        print(json.dumps(bench_device(args.rounds, args.procs, args.seed), indent=1))
    elif args.cmd == "check":
        out = causality_check(args.causality_users, panel=args.panel)
        (HERE / "check-causality.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    else:
        run(args.device, args.rounds, args.depth, args.eta, args.procs, args.seed,
            dump_preds=args.dump_preds)


if __name__ == "__main__":
    main()
