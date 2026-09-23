#!/usr/bin/env python
"""Stage 2: a causal correction on top of the probe's own prediction.

Stage 1 (`tabular-probe-20260917`) treats B as the offset and models the base residual with
four aggregate families.  Its group increments are saturated (+state -0.0009, +deck -0.0011,
whole-family stack -0.0004) and the per-row oracle over {B, FSRS-7, probe} sits 0.05 below the
best real combiner, so the remaining headroom is per-row heterogeneity.  Stage 2 keeps stage 1's
prediction as the offset (`base_margin = logit(clip(p_all))`) and adds what stage 1 never saw:

* `rt_*`  rating history -- stage 1 has no rating-derived feature at all, only outcome residuals
* `sg_*`  sub-day timing -- `elapsed_seconds`, same-session flag, previous response duration
* `dg_*`  disagreement/confidence -- probe-vs-B and probe-vs-FSRS-7 logit gaps, spread across the
          seven stage-1 variants

Everything is strictly causal: windows use rows `< t`, and `elapsed_seconds` is the gap known when
the card comes up.  Protocol copies stage 1 -- develop on the lockbox panel (96 train / 32 val /
128 eval, equal-user weights scaled to mean 1), confirm once on the fresh panel without refitting.

    scripts/capped.sh 3G .venv/bin/python evaluation/probe-stage2-20260918/stage2.py build --panel lockbox
    scripts/capped.sh 3G .venv/bin/python evaluation/probe-stage2-20260918/stage2.py build --panel fresh
    scripts/capped.sh 3G .venv/bin/python evaluation/probe-stage2-20260918/stage2.py run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / "tabular-probe-20260917"
sys.path.insert(0, str(PROBE))
import probe  # noqa: E402

CLIP = probe.CLIP
VARIANTS = ["all", "spine", "surprise", "state", "deck",
            "spine_surprise", "spine_surprise_state"]
STREAM_FEATS = ["rt_mean8", "rt_mean32", "rt_sd32", "rt_last", "rt_fail32",
                "rt_good_streak", "rt_lapse_streak", "sg_log_secs", "sg_same_session",
                "sg_prev_dur_log"]
DISAGREE_FEATS = ["dg_z_gap_b", "dg_z_gap_fsrs", "dg_var_std", "dg_var_range", "dg_n_above",
                  "dg_var_mean_minus_all"]
CARRY = ["row_index", "y", "elapsed_days", "i_raw", "lapse"]


def panel_paths(panel: str) -> tuple[Path, Path]:
    if panel == "lockbox":
        return PROBE / "features-all.parquet", HERE / "stage2-lockbox.parquet"
    return PROBE / "features-fresh.parquet", HERE / "stage2-fresh.parquet"


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def _prev(x: np.ndarray) -> np.ndarray:
    """Value of the previous row (0.0 for the first)."""
    out = np.zeros(len(x), np.float64)
    if len(x) > 1:
        out[1:] = x[:-1]
    return out


def _roll_mean_prev(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of the previous min(t, w) values of x.

    Prefix sums run over `x` itself, so the window for row t is `x[max(0, t - w):t]` and the
    denominator is `min(t, w)`.  (The first version summed `_prev(x)` and still divided by
    `min(t, w)`, i.e. it was one review stale *and* scaled by `(t-1)/t`; `serve.py` implements
    the documented semantics, which is how the parity check caught it.)

    >>> _roll_mean_prev(np.array([3.0, 3.0, 4.0]), 8)
    array([0. , 3. , 3. ])
    """
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    t = np.arange(len(x))
    n = np.minimum(t, w)
    return (c[t] - c[np.maximum(0, t - w)]) / np.maximum(1.0, n)


def _streak_prev(flags: np.ndarray) -> np.ndarray:
    """Length of the consecutive-True run ending at row t-1."""
    out = np.zeros(len(flags), np.float64)
    run = 0.0
    for t in range(len(flags)):
        out[t] = run
        run = run + 1.0 if flags[t] else 0.0
    return out


def _stage1_predictions(d: dict, names: list[str]) -> dict[str, np.ndarray]:
    """Apply the persisted stage-1 boosters (exact recipe: truncate at best_iteration)."""
    out = {}
    for v in VARIANTS:
        bst = xgb.Booster()
        bst.load_model(str(PROBE / f"model-{v}.json"))
        fnames = list(bst.feature_names or [])
        X = np.column_stack([d[f].astype(np.float32) for f in fnames])
        best = getattr(bst, "best_iteration", None)
        dm = xgb.DMatrix(X, base_margin=_logit(d["p_b"]).astype(np.float32),
                         missing=np.nan, feature_names=fnames)
        out[v] = bst.predict(
            dm, iteration_range=(0, best + 1) if best is not None else (0, 0)).astype(np.float64)
    return out


def user_table(user: int, panel: str) -> pa.Table:
    feats_pq, _ = panel_paths(panel)
    cols = ["row_index", "y", "p_fsrs", "p_b", "elapsed_days", "i_raw", "lapse"] + probe.FEATURES
    tbl = pq.read_table(feats_pq, columns=cols, filters=[("user", "==", int(user))])
    d = {c: tbl.column(c).to_numpy(zero_copy_only=False) for c in tbl.column_names}
    n = len(d["y"])
    if not n:
        raise ValueError(f"user {user}: no rows in {feats_pq.name}")

    preds = _stage1_predictions(d, probe.FEATURES)
    frame = probe.scored_frame(user)                      # rating / seconds / duration
    assert len(frame) == n, f"user {user}: frame {len(frame)} vs matrix {n}"
    assert np.array_equal(frame["y"].to_numpy(np.int64), d["y"].astype(np.int64)), user
    rating = frame["rating"].to_numpy(np.float64)
    secs = frame["elapsed_seconds"].to_numpy(np.float64)
    duration = frame["duration"].to_numpy(np.float64)

    z_all = _logit(preds["all"])
    z_var = np.column_stack([_logit(preds[v]) for v in VARIANTS])
    f: dict[str, np.ndarray] = {
        "rt_mean8": _roll_mean_prev(rating, 8),
        "rt_mean32": _roll_mean_prev(rating, 32),
        "rt_last": _prev(rating),
        "rt_fail32": _roll_mean_prev((rating < 3).astype(np.float64), 32),
        "rt_good_streak": _streak_prev(rating >= 3),
        "rt_lapse_streak": _streak_prev(rating == 1),
        "sg_log_secs": np.log1p(np.maximum(secs, 0.0)),
        "sg_same_session": (secs < 600).astype(np.float64),
        "sg_prev_dur_log": np.log1p(np.nan_to_num(_prev(duration), nan=0.0)),
        "dg_z_gap_b": z_all - _logit(d["p_b"]),
        "dg_z_gap_fsrs": z_all - _logit(d["p_fsrs"]),
        "dg_var_std": z_var.std(axis=1),
        "dg_var_range": z_var.max(axis=1) - z_var.min(axis=1),
        "dg_n_above": (z_var > 0.0).sum(axis=1).astype(np.float64),
        "dg_var_mean_minus_all": z_var.mean(axis=1) - z_all,
    }
    m32 = _roll_mean_prev(rating, 32)
    f["rt_sd32"] = np.sqrt(np.maximum(0.0, _roll_mean_prev(rating ** 2, 32) - m32 ** 2))

    out = {"user": np.full(n, user, np.int32), "p_all": preds["all"].astype(np.float32),
           "p_b": d["p_b"].astype(np.float32), "p_fsrs": d["p_fsrs"].astype(np.float32)}
    for c in CARRY:
        out[c] = d[c].astype(np.int32 if c != "y" else np.int8)
    for k in STREAM_FEATS + DISAGREE_FEATS:
        out[k] = f[k].astype(np.float32)
    return pa.table(out)


def build(panel: str, procs: int, users: list[int] | None = None) -> dict:
    feats_pq, out_pq = panel_paths(panel)
    users = users or sorted(set(pq.read_table(
        feats_pq, columns=["user"]).column("user").to_numpy().tolist()))
    t0 = time.time()
    tmp = out_pq.with_suffix(".parquet.tmp")
    writer = None
    done = 0
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for user, table in ex.map(_build_one, [(u, panel) for u in users], chunksize=1):
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            done += 1
            if done % 32 == 0 or done == len(users):
                print(f"  {done}/{len(users)} users  {time.time() - t0:.0f}s", flush=True)
    writer.close()
    tmp.replace(out_pq)
    return {"panel": panel, "users": done,
            "rows": int(pq.ParquetFile(out_pq).metadata.num_rows),
            "seconds": round(time.time() - t0, 1), "path": str(out_pq)}


def _build_one(job):
    user, panel = job
    return int(user), user_table(int(user), panel)


def _load_panel(panel: str) -> dict:
    _, path = panel_paths(panel)
    cols = ["user", "p_all", "p_b", "p_fsrs"] + CARRY + STREAM_FEATS + DISAGREE_FEATS
    tbl = pq.read_table(path, columns=cols)
    d = {c: tbl.column(c).to_numpy(zero_copy_only=False) for c in tbl.column_names}
    user = d["user"].astype(np.int64)
    d["sizes"] = {int(k): int(v) for k, v in zip(*np.unique(user, return_counts=True))}
    d["rows"] = {u: np.flatnonzero(user == u) for u in d["sizes"]}
    d["y"] = d["y"].astype(np.float64)
    return d


def _score(d: dict, p: np.ndarray, users: list[int]) -> list[dict]:
    per = []
    for u in users:
        r = d["rows"][u]
        grouping = probe.bin_grouping(np.maximum(d["elapsed_days"][r].astype(np.float64), 1e-6),
                                      np.maximum(d["i_raw"][r].astype(np.int64), 1),
                                      d["lapse"][r].astype(np.int64))
        per.append(probe.per_user_metrics(d["y"][r], p[r].astype(np.float64), grouping, len(r)))
    return per


def _fit(feats: list[str], d: dict, train: list[int], val: list[int], procs: int,
         rounds: int = 600, seed: int = 20260917) -> xgb.Booster:
    X = np.column_stack([d[f].astype(np.float32) for f in feats])
    y = d["y"].astype(np.float32)
    w = probe.equal_user_weights(d["user"], d["sizes"])
    z = _logit(d["p_all"]).astype(np.float32)
    tr = np.isin(d["user"], train)
    va = np.isin(d["user"], val)
    params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
              "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 20.0, "reg_lambda": 1.0, "eval_metric": "logloss",
              "seed": seed, "nthread": procs}
    dtr = xgb.QuantileDMatrix(X[tr], label=y[tr], weight=w[tr], base_margin=z[tr],
                              missing=np.nan, feature_names=feats)
    dva = xgb.QuantileDMatrix(X[va], label=y[va], weight=w[va], base_margin=z[va],
                              ref=dtr, missing=np.nan, feature_names=feats)
    bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "val")],
                    early_stopping_rounds=30, verbose_eval=False)
    del dtr, dva, X, y, w, z
    return bst


def _predict(bst: xgb.Booster, feats: list[str], d: dict) -> np.ndarray:
    X = np.column_stack([d[f].astype(np.float32) for f in feats])
    z = _logit(d["p_all"]).astype(np.float32)
    dm = xgb.DMatrix(X, base_margin=z, missing=np.nan, feature_names=feats)
    best = getattr(bst, "best_iteration", None)
    out = bst.predict(dm, iteration_range=(0, best + 1) if best is not None else (0, 0))
    del X, z, dm
    return out.astype(np.float64)


def run(procs: int) -> dict:
    specs = {"stream": STREAM_FEATS, "disagree": DISAGREE_FEATS,
             "all_new": STREAM_FEATS + DISAGREE_FEATS}

    lock = _load_panel("lockbox")
    split = probe.split_users(lock["sizes"])
    print(f"lockbox: {len(lock['sizes'])} users, split {[(k, len(v)) for k, v in split.items()]}")
    off = np.zeros(len(lock["y"]))
    off[:] = lock["p_all"]
    res: dict = {"specs": {k: len(v) for k, v in specs.items()},
                 "features": specs,
                 "lockbox": {"offset": {}, "variants": {}},
                 "fresh": {"offset": {}, "variants": {}},
                 "params": {"offset": "logit(clip(p_all)) stage-1 prediction", "rounds": 600,
                            "early_stopping": 30, "max_depth": 6, "eta": 0.05,
                            "min_child_weight": 20.0, "reg_lambda": 1.0,
                            "weights": "equal-user 1/n_u scaled to mean 1"}}
    per_off = _score(lock, off, split["eval"])
    res["lockbox"]["offset"] = {m: probe.equal_user(per_off, m)
                                for m in ("LogLoss", "RMSE(bins)", "AUC", "MBE")}
    res["lockbox"]["offset"]["per_user_LogLoss"] = [float(r["LogLoss"]) for r in per_off]
    print(f"offset (stage-1 `all`)   {res['lockbox']['offset']['LogLoss']:.6f}")

    models = {}
    for name, feats in specs.items():
        bst = _fit(feats, lock, split["train"], split["val"], procs)
        models[name] = bst
        p = _predict(bst, feats, lock)
        per = _score(lock, p, split["eval"])
        entry = {m: probe.equal_user(per, m) for m in ("LogLoss", "RMSE(bins)", "AUC", "MBE")}
        entry["per_user_LogLoss"] = [float(r["LogLoss"]) for r in per]
        entry["rounds"] = int(getattr(bst, "best_iteration", -1)) + 1
        entry["paired_vs_offset"] = probe.paired_bootstrap(
            np.array(entry["per_user_LogLoss"]),
            np.array(res["lockbox"]["offset"]["per_user_LogLoss"]))
        res["lockbox"]["variants"][name] = entry
        d = entry["paired_vs_offset"]
        print(f"  {name:10s} LL {entry['LogLoss']:.6f}  rounds {entry['rounds']:>4}  "
              f"Δ {d['mean']:+.6f} [{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}] "
              f"{d['improved']}/{d['worsened']}")
    del lock

    fresh = _load_panel("fresh")
    off = fresh["p_all"].astype(np.float64)
    per_off = _score(fresh, off, sorted(fresh["sizes"]))
    res["fresh"]["offset"] = {m: probe.equal_user(per_off, m)
                              for m in ("LogLoss", "RMSE(bins)", "AUC", "MBE")}
    res["fresh"]["offset"]["per_user_LogLoss"] = [float(r["LogLoss"]) for r in per_off]
    print(f"\nfresh offset (stage-1 `all`) {res['fresh']['offset']['LogLoss']:.6f}")
    for name, feats in specs.items():
        p = _predict(models[name], feats, fresh)
        per = _score(fresh, p, sorted(fresh["sizes"]))
        entry = {m: probe.equal_user(per, m) for m in ("LogLoss", "RMSE(bins)", "AUC", "MBE")}
        entry["per_user_LogLoss"] = [float(r["LogLoss"]) for r in per]
        entry["paired_vs_offset"] = probe.paired_bootstrap(
            np.array(entry["per_user_LogLoss"]),
            np.array(res["fresh"]["offset"]["per_user_LogLoss"]))
        res["fresh"]["variants"][name] = entry
        d = entry["paired_vs_offset"]
        print(f"  {name:10s} LL {entry['LogLoss']:.6f}  Δ {d['mean']:+.6f} "
              f"[{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}] {d['improved']}/{d['worsened']}")

    (HERE / "results.json").write_text(json.dumps(res, indent=1, allow_nan=False))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--panel", default="lockbox", choices=["lockbox", "fresh"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--users", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.panel, args.procs, args.users), indent=1))
    else:
        print(json.dumps(run(args.procs), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
