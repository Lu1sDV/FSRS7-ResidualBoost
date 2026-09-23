#!/usr/bin/env python
"""Nested (train / validation / held) pipeline for the 42-feature probe+stream leader.

Public API (consumed by ``run.py``; behaviour pinned by ``test_submission.py``):

    make_splits(users, seed)             -> two outer {train, val, held} fold dicts
    build_features(cache, beta)          -> (X (n, 42) float32, p_b float64)
    fit_calibration(users, loader)       -> equal-user 3-coefficient shrinkage fit
    fit_outer(train, val, loader, ...)   -> booster + calibration lineage on disk
    predict_user(cache, beta, booster)   -> probabilities from the saved booster

Everything is train-only.  The calibration coefficients are fitted on training users
only, inner 3-group cross-fitted so a training user never sees its own label; the
booster is trained with equal-user (1/n_u scaled to mean 1) weights and 42 causal
features on base_margin logit(p_b); no ``held`` user is ever handed to ``loader``.

Feature code is imported from the historical probe/stage2 folders through a
``sys.path`` shim; no stored coefficient file (crossfit betas, saved boosters) is read.

Caching strategy (memory bound, no corpus-sized matrix):

    pass 1  train users -> per-user calibration columns (.npy, memmapped, streamed)
    pass 2  train+val users -> per-user 42-column feature .npy + (y, logit p_b)
    pass 3  XGBoost ExtMemQuantileDMatrix(DataIter) over those .npy, disk cache,
            one user per batch, equal-user weights scaled globally (never per batch)

The private feature/calibration temporaries live under ``<outdir>/.nested-work-*`` and
are removed when the fit returns (or fails).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.special import expit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_PROBE_DIR = REPO / "evaluation" / "tabular-probe-20260917"
_STAGE2_DIR = REPO / "evaluation" / "probe-stage2-20260918"
for _p in (_PROBE_DIR, _STAGE2_DIR):  # the only sys.path manipulation
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import probe
import stage2 as s2

# 32 probe features + the 10 stage-2 `stream` features = the leader's 42 causal columns.
FEATURE_NAMES = list(probe.FEATURES) + list(s2.STREAM_FEATS)
assert len(FEATURE_NAMES) == 42, len(FEATURE_NAMES)
assert len(set(FEATURE_NAMES)) == 42

SEED = 20260919  # split shuffle
SEED_BOOSTER = 20260917  # xgboost seed (matches the historical leader)
N_INNER = 3  # inner calibration cross-fit groups
VAL_FRACTION = 0.15
MIN_TRAIN = N_INNER  # one user per inner group at least
CHUNK = 1 << 20  # calibration streaming chunk (rows)

BOOSTER_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20.0,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "seed": SEED_BOOSTER,
}


# ------------------------------------------------------------------ splits


def _sorted_users(users) -> list[int]:
    return sorted({int(u) for u in users})


def make_splits(users, seed: int = SEED) -> list[dict]:
    """Two outer folds covering every user exactly once as `held`.

    Users are sorted first, so the result does not depend on input order.  Within an
    outer fold the held-out half is taken first, then the remaining half is split with
    the same seed-fixed shuffle into ~15% validation and the rest training.
    """
    us = _sorted_users(users)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(us))
    half = len(us) // 2
    halves = (order[:half], order[half:])
    folds = []
    for k, held_idx in enumerate(halves):
        rest = rng.permutation(halves[1 - k])  # the other outer half
        n_rest = len(rest)
        if n_rest < MIN_TRAIN + 1:
            raise ValueError(
                f"fold {k}: {n_rest} non-held users, need >= {MIN_TRAIN + 1}"
            )
        n_val = max(1, round(VAL_FRACTION * n_rest))
        n_val = min(n_val, n_rest - MIN_TRAIN)
        folds.append(
            {
                "train": sorted(us[i] for i in rest[n_val:]),
                "val": sorted(us[i] for i in rest[:n_val]),
                "held": sorted(us[i] for i in held_idx),
            }
        )
    return folds


def inner_groups(users, n: int = N_INNER, seed: int = SEED + 1) -> list[list[int]]:
    """Deterministic near-equal partition of `users` for cross-fitted calibration."""
    us = _sorted_users(users)
    if len(us) < n:
        raise ValueError(f"inner_groups: {len(us)} users < {n} groups")
    perm = np.random.default_rng(seed).permutation(len(us))
    return [sorted(us[i] for i in part) for part in np.array_split(perm, n)]


# ------------------------------------------------------------------ features


def _stream_features(rating, secs, duration) -> dict[str, np.ndarray]:
    """The 10 `stream` columns, exactly as stage2.user_table computes them."""
    r = np.asarray(rating, np.float64)
    s = np.asarray(secs, np.float64)
    du = np.asarray(duration, np.float64)
    m32 = s2._roll_mean_prev(r, 32)
    return {
        "rt_mean8": s2._roll_mean_prev(r, 8),
        "rt_mean32": m32,
        "rt_sd32": np.sqrt(np.maximum(0.0, s2._roll_mean_prev(r**2, 32) - m32**2)),
        "rt_last": s2._prev(r),
        "rt_fail32": s2._roll_mean_prev((r < 3).astype(np.float64), 32),
        "rt_good_streak": s2._streak_prev(r >= 3),
        "rt_lapse_streak": s2._streak_prev(r == 1),
        "sg_log_secs": np.log1p(np.maximum(s, 0.0)),
        "sg_same_session": (s < 600).astype(np.float64),
        "sg_prev_dur_log": np.log1p(np.nan_to_num(s2._prev(du), nan=0.0)),
    }


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), probe.CLIP, 1.0 - probe.CLIP)
    return np.log(p / (1.0 - p))


def _feature_dict(cache) -> dict[str, np.ndarray]:
    """Metric-free row dict for one user, in the contract's cache field names."""
    y = np.asarray(cache["y"], np.float64)
    p_fsrs = np.asarray(cache["p_fsrs"], np.float64)
    n = len(y)
    if n == 0:
        raise ValueError("empty cache")
    for k in (
        "i",
        "elapsed_days",
        "lapse",
        "nth_today",
        "day_offset",
        "card_id",
        "deck_id",
        "note_id",
        "rating",
        "elapsed_seconds",
        "duration",
    ):
        if k not in cache:
            raise KeyError(f"cache missing contract field {k!r}")
    return {
        "y": y,
        "p_fsrs": p_fsrs,
        "i": np.asarray(cache["i"], np.int64),
        "elapsed": np.asarray(cache["elapsed_days"], np.float64),
        "lapse": np.asarray(cache["lapse"], np.int64),
        "nth_today": np.asarray(cache["nth_today"], np.float64),
        "day": np.asarray(cache["day_offset"], np.float64),
        "rating": np.asarray(cache["rating"], np.float64),
        "secs": np.asarray(cache["elapsed_seconds"], np.float64),
        "duration": np.asarray(cache["duration"], np.float64),
        "card_code": probe._codes(np.asarray(cache["card_id"])),
        "deck_code": probe._codes(np.asarray(cache["deck_id"])),
        "note_code": probe._codes(np.asarray(cache["note_id"])),
    }


def build_features(cache, beta) -> tuple[np.ndarray, np.ndarray]:
    """42 causal features plus the calibrated base probability for one user's cache.

    Row t uses rows < t only, so perturbing a later label/rating/duration cannot change
    any earlier row.  ``beta`` are the three shrinkage coefficients of `derive_p_b`.
    """
    d = _feature_dict(cache)
    beta = np.asarray(beta, np.float64).reshape(-1)
    if beta.shape != (3,):
        raise ValueError(f"beta must have 3 coefficients, got {beta.shape}")
    p_b = probe.derive_p_b(d["y"], d["p_fsrs"], beta)
    f = probe.compute_features(
        d["y"],
        p_b,
        d["p_fsrs"],
        d["i"],
        d["elapsed"],
        d["lapse"],
        d["nth_today"],
        d["day"],
        d["card_code"],
        d["deck_code"],
        d["note_code"],
    )
    f.update(_stream_features(d["rating"], d["secs"], d["duration"]))
    X = np.empty((len(d["y"]), 42), np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        X[:, j] = f[name]
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature value")
    return X, np.asarray(p_b, np.float64)


# ------------------------------------------------------------------ calibration


def _calib_array(cache) -> np.ndarray:
    """Per-row [logit p^F, x16, x256, y] -- everything the 3-coefficient fit needs."""
    y = np.asarray(cache["y"], np.float64)
    p_fsrs = np.asarray(cache["p_fsrs"], np.float64)
    x16, x256 = probe.shrunk_base_windows(y, p_fsrs)
    return np.column_stack([_logit(p_fsrs), x16, x256, y])


def _calib_columns(workdir, users, loader) -> tuple[dict[int, Path], dict[int, int]]:
    paths, sizes = {}, {}
    for u in users:
        cols = _calib_array(loader(u))
        if not len(cols):
            raise ValueError(f"user {u}: empty cache")
        p = Path(workdir) / f"calib-{u}.npy"
        np.save(p, cols)
        paths[u], sizes[u] = p, len(cols)
    return paths, sizes


def _fit_columns(
    paths: dict[int, Path],
    sizes: dict[int, int],
    *,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict:
    """Fit the equal-user logistic objective with a safeguarded standard solver."""
    from scipy.optimize import minimize

    users = sorted(paths)
    count = len(users)
    weights = {u: 1.0 / (count * sizes[u]) for u in users}
    design = np.empty((CHUNK, 3), np.float64)

    def objective(theta):
        gradient = np.zeros(3)
        loss = 0.0
        for user in users:
            rows = np.load(paths[user], mmap_mode="r")
            for start in range(0, sizes[user], CHUNK):
                block = rows[start : start + CHUNK]
                x = design[: len(block)]
                x[:, 0] = 1.0
                x[:, 1:] = block[:, 1:3]
                z = block[:, 0] + x @ theta
                y = block[:, 3]
                gradient += weights[user] * (x.T @ (expit(z) - y))
                loss += weights[user] * float(np.sum(np.logaddexp(0.0, z) - y * z))
            del rows
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(3),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-12, "maxls": 40},
    )
    if (
        not fitted.success
        or not np.isfinite(fitted.x).all()
        or not np.isfinite(fitted.fun)
    ):
        raise RuntimeError(f"calibration failed: {fitted.message}")
    return {
        "beta": fitted.x.tolist(),
        "objective": float(fitted.fun),
        "iterations": int(fitted.nit),
        "users": count,
        "rows": int(sum(sizes.values())),
    }


def fit_calibration(
    users, loader, *, workdir=None, max_iter: int = 100, tol: float = 1e-10
) -> dict:
    """Fit the 3 shrinkage coefficients on `users` only, from `loader` caches.

    Returns {"beta": [b0, b1, b2], "objective", "iterations", "users", "rows"}.
    """
    us = _sorted_users(users)
    if not us:
        raise ValueError("fit_calibration: no users")
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="nested-calib-") as td:
            paths, sizes = _calib_columns(td, us, loader)
            return _fit_columns(paths, sizes, max_iter=max_iter, tol=tol)
    paths, sizes = _calib_columns(workdir, us, loader)
    try:
        return _fit_columns(paths, sizes, max_iter=max_iter, tol=tol)
    finally:
        for p in paths.values():
            p.unlink(missing_ok=True)


# ------------------------------------------------------------------ outer fit


class _CheckpointEarlyStopping(xgb.callback.TrainingCallback):
    """Keep the validation best and patience across a crashed booster fit."""

    def __init__(self, outdir, patience):
        self.path = Path(outdir) / "boost-checkpoint.pkl"
        self.patience = patience

        self.path.parent.mkdir(parents=True, exist_ok=True)

    def before_training(self, model):
        self.start = model.num_boosted_rounds()
        score = model.attr("best_score")
        if self.start and score is None:
            raise ValueError("checkpoint missing early-stopping state")
        self.best = float(score) if score is not None else None
        self.best_round = int(model.attr("best_iteration")) if score is not None else -1
        return model

    def after_iteration(self, model, epoch, evals_log):
        epoch += self.start
        score = float(evals_log["val"]["logloss"][-1])
        if self.best is None or score < self.best:
            self.best, self.best_round = score, epoch
            model.set_attr(best_score=str(score), best_iteration=str(epoch))
        if (epoch + 1) % 10 == 0:
            temporary = self.path.with_name(".boost-checkpoint.tmp.pkl")
            temporary.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
            os.replace(temporary, self.path)
            print(f"checkpoint round={epoch + 1}", flush=True)
        return epoch - self.best_round >= self.patience


def _equal_user_weights(sizes: dict[int, int], users: list[int]) -> dict[int, float]:
    """1/n_u scaled so the mean weight over these rows is exactly 1."""
    n_tot = sum(sizes[u] for u in users)
    return {u: (n_tot / len(users)) / sizes[u] for u in users}


class _FeatureIter(xgb.DataIter):
    """One user per batch from the per-user feature .npy files (external memory)."""

    def __init__(self, users, featdir, weights, cache_prefix):
        super().__init__(cache_prefix=str(cache_prefix))
        self.users = list(users)
        self.featdir = Path(featdir)
        self.weights = weights
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def next(self, input_data) -> bool:
        if self.i >= len(self.users):
            return False
        u = self.users[self.i]
        self.i += 1
        X = np.load(self.featdir / f"X-{u}.npy")
        meta = np.load(self.featdir / f"M-{u}.npy")
        input_data(
            data=X,
            label=meta[:, 0],
            weight=np.full(len(X), self.weights[u], np.float32),
            base_margin=meta[:, 1],
            feature_names=FEATURE_NAMES,
        )
        return True


def _write_user_features(workdir, u, cache, beta) -> int:
    """Build one user's features once and persist them for the (multi-pass) iterator."""
    X, p_b = build_features(cache, beta)
    np.save(Path(workdir) / f"X-{u}.npy", X)
    np.save(
        Path(workdir) / f"M-{u}.npy",
        np.column_stack([np.asarray(cache["y"], np.float64), _logit(p_b)]).astype(
            np.float32
        ),
    )
    return len(X)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_or_none(v):
    return float(v) if v is not None and np.isfinite(v) else None


def _lineage(params: dict, splits: dict) -> dict:
    return {
        "files": {
            p.name: _sha256(p)
            for p in (
                Path(__file__),
                _PROBE_DIR / "probe.py",
                _STAGE2_DIR / "stage2.py",
            )
        },
        "packages": {
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "features": FEATURE_NAMES,
        "params": params,
        "splits": splits,
        "offset": "logit(clip(p_b)) with beta fitted on training users only",
    }


def fit_outer(
    train,
    val,
    loader,
    outdir,
    rounds: int = 600,
    threads: int = 4,
    *,
    seed: int = SEED_BOOSTER,
    n_inner: int = N_INNER,
    early_stopping_rounds: int = 30,
    max_iter: int = 100,
    checkpoint_key: str | None = None,
) -> dict:
    """Fit the leader on `train`, early-stopping on `val`; never touch `held`.

    Returns a dict with the model path, the calibration coefficients applied to
    validation/held users, every calibration fit's provenance (fit_users/apply_users)
    and the early-stopping best_iteration.
    """
    t0 = time.time()
    train, val = _sorted_users(train), _sorted_users(val)
    if not train or not val:
        raise ValueError("fit_outer: train and val must be non-empty")
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"fit_outer: {len(overlap)} users in both train and val")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "boost-checkpoint.pkl"
    if checkpoint_key is not None:
        identity = {
            "key": checkpoint_key,
            "train": train,
            "val": val,
            "rounds": rounds,
            "threads": threads,
            "seed": seed,
            "inner": n_inner,
            "patience": early_stopping_rounds,
        }
        identity_path = outdir / "boost-checkpoint.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("checkpoint inputs or fit settings differ")
        elif checkpoint.exists():
            raise ValueError("checkpoint missing input identity")
        else:
            temporary = outdir / ".boost-checkpoint.json.tmp"
            temporary.write_text(json.dumps(identity, sort_keys=True))
            os.replace(temporary, identity_path)
    work = Path(tempfile.mkdtemp(prefix=".nested-work-", dir=outdir))
    params = dict(BOOSTER_PARAMS, seed=seed, nthread=threads, max_depth=6)
    try:
        # ---- pass 1: calibration columns, then the inner cross-fits
        paths, sizes = _calib_columns(work, train, loader)
        groups = inner_groups(train, n=n_inner)
        calibration_fits, beta_of = [], {}
        for gi, group in enumerate(groups):
            keep = [u for u in train if u not in set(group)]
            fit = _fit_columns(
                {u: paths[u] for u in keep},
                {u: sizes[u] for u in keep},
                max_iter=max_iter,
            )
            calibration_fits.append(
                {
                    "fold": f"inner{gi}",
                    "fit_users": keep,
                    "apply_users": list(group),
                    **fit,
                }
            )
            for u in group:
                beta_of[u] = fit["beta"]
        full = _fit_columns(paths, sizes, max_iter=max_iter)
        calibration_fits.append(
            {"fold": "val", "fit_users": list(train), "apply_users": list(val), **full}
        )
        beta_train = full["beta"]
        for p in paths.values():
            p.unlink(missing_ok=True)

        # ---- pass 2: per-user features, cross-fitted for train, train-fitted for val
        tr_sizes = {
            u: _write_user_features(work, u, loader(u), beta_of[u]) for u in train
        }
        va_sizes = {
            u: _write_user_features(work, u, loader(u), beta_train) for u in val
        }

        # ---- pass 3: external-memory training, one user per batch
        dtr = xgb.ExtMemQuantileDMatrix(
            _FeatureIter(
                train, work, _equal_user_weights(tr_sizes, train), work / "xgb-train"
            ),
            missing=np.nan,
            nthread=threads,
        )
        dva = xgb.ExtMemQuantileDMatrix(
            _FeatureIter(
                val, work, _equal_user_weights(va_sizes, val), work / "xgb-val"
            ),
            ref=dtr,
            missing=np.nan,
            nthread=threads,
        )
        checkpoint_model = (
            pickle.loads(checkpoint.read_bytes())
            if checkpoint_key is not None and checkpoint.exists()
            else None
        )
        completed = checkpoint_model.num_boosted_rounds() if checkpoint_model else 0
        if completed > rounds:
            raise ValueError("checkpoint has more rounds than requested")
        if completed == rounds and checkpoint_model is not None:
            bst = checkpoint_model
        else:
            bst = xgb.train(
                params,
                dtr,
                num_boost_round=int(rounds) - completed,
                evals=[(dva, "val")],
                xgb_model=checkpoint_model,
                callbacks=(
                    [_CheckpointEarlyStopping(outdir, early_stopping_rounds)]
                    if checkpoint_key is not None
                    else None
                ),
                early_stopping_rounds=(
                    None if checkpoint_key is not None else int(early_stopping_rounds)
                ),
                verbose_eval=10 if checkpoint_key is not None else False,
            )
        best = int(getattr(bst, "best_iteration", int(rounds) - 1))
        del dtr, dva

        # ---- artifacts (temp file + replace, so a crash never leaves a half artifact)
        model = outdir / "model.json"
        tmp = outdir / ".model.tmp.json"  # .json -> JSON, not UBJSON default
        bst.save_model(str(tmp))
        os.replace(tmp, model)
        calib = {
            "beta": beta_train,
            "fits": calibration_fits,
            "inner_groups": groups,
            "best_iteration": best,
            "resumed_rounds": completed,
            "params": {k: v for k, v in params.items()},
            "features": FEATURE_NAMES,
            "rounds": int(rounds),
            "early_stopping_rounds": int(early_stopping_rounds),
            "train_users": train,
            "val_users": val,
            "train_rows": int(sum(tr_sizes.values())),
            "val_rows": int(sum(va_sizes.values())),
            "weighting": "equal-user 1/n_u scaled to mean 1 per matrix",
            "val_logloss_best": _finite_or_none(getattr(bst, "best_score", None)),
            "seconds": round(time.time() - t0, 1),
        }
        calib_path = outdir / "calibration.json"
        tmp = outdir / ".calibration.json.tmp"
        tmp.write_text(json.dumps(calib, indent=1, allow_nan=False))
        os.replace(tmp, calib_path)
        lin = _lineage(params, {"train": train, "val": val})
        tmp = outdir / ".lineage.json.tmp"
        tmp.write_text(json.dumps(lin, indent=1, allow_nan=False))
        os.replace(tmp, outdir / "lineage.json")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return {
        "model": str(model.resolve()),
        "beta": beta_train,
        "calibration_fits": calibration_fits,
        "best_iteration": best,
        "resumed_rounds": completed,
        "inner_groups": groups,
        "params": {k: v for k, v in params.items()},
        "features": FEATURE_NAMES,
        "calibration_json": str(calib_path.resolve()),
        "lineage_json": str((outdir / "lineage.json").resolve()),
        "train_users": train,
        "val_users": val,
        "train_rows": int(sum(tr_sizes.values())),
        "val_rows": int(sum(va_sizes.values())),
        "seconds": round(time.time() - t0, 1),
    }


# ------------------------------------------------------------------ inference


def predict_user(cache, beta, booster, threads: int = 4) -> np.ndarray:
    """Posterior probabilities for one user, truncating at the model's best_iteration.

    ``booster`` is an ``xgb.Booster`` or a path to a saved model.
    """
    if not isinstance(booster, xgb.Booster):
        bst = xgb.Booster()
        bst.load_model(str(booster))
        booster = bst
    X, p_b = build_features(cache, beta)
    z = _logit(p_b).astype(np.float32)
    dm = xgb.DMatrix(
        X, base_margin=z, missing=np.nan, feature_names=FEATURE_NAMES, nthread=threads
    )
    best = getattr(booster, "best_iteration", None)
    rng = (0, int(best) + 1) if best is not None else (0, booster.num_boosted_rounds())
    return np.asarray(booster.predict(dm, iteration_range=rng), np.float64)
