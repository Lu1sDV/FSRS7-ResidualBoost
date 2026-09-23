#!/usr/bin/env python
"""Pinned-upstream baseline runner: official per-user predictions + keyed NPZ cache.

Runs the *current* upstream checkout (``evaluation/submission-20260919/upstream``,
commit ``bd9110f791e5b37282c55a9aa8db35f68f0c4aa2``) **unmodified** through its own
official per-user entry point ``script.process`` with the official flags

    --algo FSRS-7 --short --secs --recency --equalize_test_with_non_secs --file

(``--file`` is what makes upstream call its persistence hook; ``--short/--secs/--recency/
--equalize_test_with_non_secs/--algo FSRS-7`` is the official protocol whose result file
upstream calls ``FSRS-7-short-secs-recency-equalize_test_with_non_secs``).  The ONLY thing
monkeypatched is ``save_evaluation_file``: its persistence is replaced by an in-memory
capture of the per-review evaluation frame, so nothing the runner does can change any
prediction or metric.  The capture is written as an atomic per-user ``<user>.npz`` cache
plus a ``<user>.json`` sidecar keyed by a content identity.

Everything else about upstream is untouched: its own ``parse_args`` (strict) sees only the
official argv, which this module builds itself and never mixes with runner arguments.

CLI
---
    baseline.py --upstream U --data D --out O --users users.json [--workers 1|2]
    baseline.py --upstream U --data D --out O --all [--workers 1|2]
    baseline.py --upstream U --data D --out O --user 42        # one user (child mode)

``--users`` takes either a JSON list of ids or a JSON object with a ``user_ids`` list.
``--all`` scans ``<data>/revlogs/user_id=*/``.  ``--dataset-revision REV`` (free-form
string) is recorded in every sidecar and is part of the per-user identity; the caller
supplies it.  Each user runs in a **fresh child process** (memory released between users,
one user in memory at a time), with ``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`` and the GPU
hidden.  The child interpreter is ``<upstream>/.venv/bin/python`` when it exists, else
``sys.executable``; ``--python PATH`` overrides it.

Exit status is 0 only when every requested user is either cached or a recorded exclusion;
any unexpected failure is reported and yields status 1.  Progress is printed per user, and
finished users are skipped on resume.

Cache contract
--------------
``<out>/<user>.npz`` holds exactly these arrays, all the same length, in upstream scored
row order (fold-major over the five ``equalize_test_with_non_secs`` test folds, which is
also upstream's review_th order)::

    y, p_fsrs, review_th, fold, i, elapsed_days, lapse, nth_today, day_offset,
    card_id, deck_id, note_id, rating, elapsed_seconds, duration

``y``/``p_fsrs`` are float64 (upstream scored label and FSRS-7 probability);
``elapsed_days`` is the upstream day interval (float64); ``lapse`` is upstream's
``rmse_bins_lapse`` (int64); ``fold`` is int8 0..4 from the boolean ``{k}_test`` masks.
``deck_id``/``note_id`` come from ``<data>/cards`` per user; cards absent from that file
(``~5%`` of card ids, deleted cards) get the unique synthetic id ``-(card_id + 1)`` in both
columns, which is exactly ``probe.scored_frame``'s convention and invents no sibling link.
``review_th`` must be strictly increasing; folds must be non-decreasing and cover 0..4.

``<out>/<user>.json`` records the identity digest, the identity fields (upstream commit +
sha256 over the ``git ls-files`` .py sources + capture-wrapper hash + runtime versions +
official flags + dataset revision + sha256 of every raw input file used),
``cache_sha256``, ``stats`` (upstream metrics/parameters), and elapsed seconds.

Resume refuses a sidecar whose identity fields differ from the current run (and refuses an
npz with no sidecar at all: legacy caches are never reused) instead of silently mixing
sources; ``--force`` re-runs the user and overwrites.

``load_user(cache_dir, user)`` validates sidecar + sha256 + array names/dtypes/shape +
chronology and returns the loaded array dict (this is the only accessor nested stages use).
``run_user(upstream, data, out, user)`` is the single-user entry point used by run.py's
smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_UPSTREAM = HERE / "upstream"
N_SPLITS = 5

# The official protocol flags.  `--file` only enables upstream's persistence hook (which we
# replace with an in-memory capture); it cannot change a prediction or a metric.
OFFICIAL_FLAGS = [
    "--algo",
    "FSRS-7",
    "--short",
    "--secs",
    "--recency",
    "--equalize_test_with_non_secs",
    "--file",
    "--torch_num_threads",
    "1",
]

# Cache array name -> required dtype.  Contract with nested.py / the stage-2 features.
CACHE_SPEC = {
    "y": "float64",
    "p_fsrs": "float64",
    "review_th": "int64",
    "fold": "int8",
    "i": "int64",
    "elapsed_days": "float64",
    "lapse": "int64",
    "nth_today": "int64",
    "day_offset": "int64",
    "card_id": "int64",
    "deck_id": "int64",
    "note_id": "int64",
    "rating": "float64",
    "elapsed_seconds": "float64",
    "duration": "float64",
}

# Upstream's own explicit "this user cannot be evaluated" signal, raised by
# features/base.py's postprocessing when no review survives the non-secs filters.  User 4371
# is the corpus case.  Matched exactly (type + message), so no other failure is ever turned
# into an exclusion.
NO_ELIGIBLE_DATA = (
    "ValueError",
    "No data after handling outliers and non-continuous rows",
)


class CacheError(RuntimeError):
    """Cache/identity problem: refuse rather than reuse or overwrite silently."""


# --------------------------------------------------------------------- hashing helpers


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_input_unit(path: Path) -> dict | None:
    """Content identity of one input unit (a parquet file, or a hive partition dir)."""
    if not path.exists():
        return None
    if path.is_file():
        files = [path]
        base = path.parent
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file())
        base = path
    h = hashlib.sha256()
    names, total = [], 0
    for f in files:
        rel = f.relative_to(base).as_posix()
        size = f.stat().st_size
        digest = _sha256_file(f)
        h.update(f"{rel}\0{digest}\0{size}\0".encode())
        names.append(rel)
        total += size
    return {"sha256": h.hexdigest(), "bytes": total, "files": names}


def data_files(data: Path, user: int) -> dict:
    """sha256 of every raw input file upstream reads for this user."""
    return {
        kind: hash_input_unit(data / kind / f"user_id={user}")
        for kind in ("revlogs", "cards")
    }


def _git(upstream: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(upstream), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {upstream}: {proc.stderr.strip()}"
        )
    return proc.stdout


_SOURCE_CACHE: dict[str, dict] = {}


def upstream_source(upstream: Path) -> dict:
    """Commit + sha256 over the tracked .py sources (git ls-files, so .venv/.git stay out)."""
    key = str(upstream)
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = _upstream_source(upstream)
    return _SOURCE_CACHE[key]


def _upstream_source(upstream: Path) -> dict:
    """sha256 over the tracked .py sources present on disk.

    ``git ls-files`` is the file list (so installed ``.venv``/``.git`` never enter the
    hash); the checkout is sparse, so files tracked-but-not-checked-out are recorded
    explicitly instead of being silently ignored.
    """
    rels = [
        line for line in _git(upstream, "ls-files", "--", "*.py").splitlines() if line
    ]
    if not rels:
        raise RuntimeError(f"no tracked .py sources under {upstream}")
    commit = _git(upstream, "rev-parse", "HEAD").strip()
    h = hashlib.sha256()
    total, present, absent = 0, 0, []
    for rel in sorted(rels):
        path = upstream / rel
        if not path.is_file():
            absent.append(rel)
            continue
        size = path.stat().st_size
        h.update(f"{rel}\0{_sha256_file(path)}\0{size}\0".encode())
        total += size
        present += 1
    if not present:
        raise RuntimeError(f"no checked-out .py sources under {upstream}")
    h.update(("absent\0" + "\0".join(absent) + "\0").encode())
    return {
        "commit": commit or None,
        "file_list": "git ls-files -- '*.py'",
        "n_tracked": len(rels),
        "n_present": present,
        "absent": absent,
        "bytes": total,
        "sha256": h.hexdigest(),
    }


def identity_fields(
    upstream: Path,
    flags: list[str],
    dataset_revision: str | None,
    user: int,
    files: dict,
    runtime: dict | None = None,
) -> dict:
    """Everything a cache must have been produced from.  Content-keyed (no paths)."""
    src = upstream_source(upstream)
    return {
        "upstream_commit": src["commit"],
        "upstream_source_sha256": src["sha256"],
        "upstream_source_files": src["n_present"],
        "capture_sha256": _sha256_file(Path(__file__)),
        "runtime_versions": env_versions() if runtime is None else runtime,
        "flags": list(flags),
        "dataset_revision": dataset_revision,
        "user": int(user),
        "data_files": files,
    }


def identity_digest(fields: dict) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def env_versions() -> dict:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for pkg in (
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "relplot",
        "scipy",
        "scikit-learn",
        "fsrs-optimizer",
    ):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _jsonable(obj):
    """Undefined metrics become JSON null; numpy scalars become plain numbers."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):
        return _jsonable(obj.item())
    return obj


# ------------------------------------------------------------------------ capture frame


def _test_mask(frame, split: int):
    name = f"{split}_test"
    if name not in frame.columns:
        raise RuntimeError(f"captured frame has no fold mask column {name!r}")
    return frame[name].to_numpy(dtype=bool)


def cache_arrays(frame, cards_path: Path) -> dict:
    """The 15 contract arrays, in frame order, with every invariant asserted."""
    import numpy as np

    n = len(frame)
    if n == 0:
        raise RuntimeError("captured evaluation frame is empty (no scored rows)")

    counts = np.zeros(n, dtype=np.int8)
    fold = np.zeros(n, dtype=np.int8)
    for k in range(N_SPLITS):
        mask = _test_mask(frame, k).astype(np.int8)
        counts += mask
        fold += k * mask
    if not (counts == 1).all():
        raise RuntimeError("fold masks are not a partition of the scored rows")
    if set(np.unique(fold).tolist()) != set(range(N_SPLITS)):
        raise RuntimeError(
            f"fold values {np.unique(fold).tolist()} != 0..{N_SPLITS - 1}"
        )

    review_th = frame["review_th"].to_numpy(np.int64)
    if not (np.diff(review_th) > 0).all():
        raise RuntimeError("review_th is not strictly increasing in captured order")

    y = frame["y"].to_numpy(np.float64)
    if not np.isin(y, (0.0, 1.0)).all():
        raise RuntimeError("labels y are not binary")
    p_fsrs = frame["p"].to_numpy(np.float64)
    if not (np.isfinite(p_fsrs).all() and ((p_fsrs >= 0) & (p_fsrs <= 1)).all()):
        raise RuntimeError("upstream probabilities outside [0, 1] or non-finite")

    card_id = frame["card_id"].to_numpy(np.int64)
    deck_id, note_id = _card_ids(card_id, cards_path)

    lapse = frame["rmse_bins_lapse"].to_numpy(np.int64)
    if (lapse < 0).any():
        raise RuntimeError("negative rmse_bins_lapse")
    elapsed_days = frame["elapsed_days"].to_numpy(np.float64)
    if (elapsed_days < 0).any():
        raise RuntimeError("negative elapsed_days")

    return {
        "y": y,
        "p_fsrs": p_fsrs,
        "review_th": review_th,
        "fold": fold,
        "i": frame["i"].to_numpy(np.int64),
        "elapsed_days": elapsed_days,
        "lapse": lapse,
        "nth_today": frame["nth_today"].to_numpy(np.int64),
        "day_offset": frame["day_offset"].to_numpy(np.int64),
        "card_id": card_id,
        "deck_id": deck_id,
        "note_id": note_id,
        "rating": frame["rating"].to_numpy(np.float64),
        "elapsed_seconds": frame["elapsed_seconds"].to_numpy(np.float64),
        "duration": frame["duration"].to_numpy(np.float64),
    }


def _card_ids(card_id, cards_path: Path):
    """deck_id/note_id per scored row; missing card rows get -(card_id+1) in both."""
    import numpy as np
    import pandas as pd

    if cards_path.exists():
        cards = pd.read_parquet(cards_path, columns=["card_id", "note_id", "deck_id"])
        cards = cards.drop_duplicates("card_id").set_index("card_id")
        deck = cards["deck_id"].reindex(card_id).to_numpy(np.float64).copy()
        note = cards["note_id"].reindex(card_id).to_numpy(np.float64).copy()
    else:
        deck = np.full(len(card_id), np.nan)
        note = np.full(len(card_id), np.nan)

    missing = ~np.isfinite(deck) | ~np.isfinite(note)
    if missing.any():
        syn = -(card_id.astype(np.int64) + 1)
        deck[missing] = syn[missing]
        note[missing] = syn[missing]
    if not (np.isfinite(deck).all() and np.isfinite(note).all()):
        raise RuntimeError("unmapped card_id after synthetic deck/note assignment")
    return deck.astype(np.int64), note.astype(np.int64)


# ------------------------------------------------------------------ upstream (child)


def _write_npz(path: Path, arrays: dict) -> None:
    import numpy as np

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _capture_user(
    upstream: Path, data: Path, out: Path, user: int, dataset_revision: str | None
) -> dict:
    """Child mode: import upstream, run its official process(), write cache + sidecar."""
    started = time.monotonic()
    argv = [str(upstream / "script.py"), *OFFICIAL_FLAGS, "--data", str(data)]
    sys.path.insert(0, str(upstream))
    sys.argv = argv  # upstream parses THIS, never the runner's arguments

    import model_processors
    import script
    import utils

    captured = []

    def capture(user_id, df, config):  # replaces utils.save_evaluation_file only
        captured.append((user_id, df))

    script.save_evaluation_file = capture
    utils.save_evaluation_file = capture
    model_processors.save_evaluation_file = capture

    result, error = script.process(user)
    elapsed = round(time.monotonic() - started, 1)

    files = data_files(data, user)
    fields = identity_fields(upstream, OFFICIAL_FLAGS, dataset_revision, user, files)
    sidecar: dict = {
        "user": int(user),
        "rows": None,
        "cache": None,
        "cache_sha256": None,
        "stats": None,
        "status": "failed",
        "excluded_reason": None,
        "error": None,
        "elapsed_seconds": elapsed,
        "interpreter": {"path": sys.executable, "version": sys.version.split()[0]},
        "versions": env_versions(),
        "identity": identity_digest(fields),
        "identity_fields": fields,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if error is not None:
        kind = _error_kind(error)
        if kind is None or user != 4371:
            sidecar["error"] = error
            return sidecar
        sidecar["status"] = "excluded"
        sidecar["excluded_reason"] = kind
        sidecar["error"] = error
        _write_sidecar(out, user, sidecar)
        return sidecar

    if len(captured) != 1:
        raise RuntimeError(
            f"expected exactly one save_evaluation_file call, got {len(captured)}"
        )
    captured_user, frame = captured[0]
    if captured_user != user:
        raise RuntimeError(f"upstream captured user {captured_user}, expected {user}")

    arrays = cache_arrays(frame, data / "cards" / f"user_id={user}")
    npz_path = out / f"{user}.npz"
    _write_npz(npz_path, arrays)
    sidecar["status"] = "ok"
    sidecar["rows"] = len(frame)
    sidecar["cache"] = npz_path.name
    sidecar["cache_sha256"] = _sha256_file(npz_path)
    sidecar["stats"] = _jsonable(result[0] if result else None)
    if sidecar["stats"] is None:
        raise RuntimeError("upstream returned no stats for a successful user")
    _write_sidecar(out, user, sidecar)
    return sidecar


def _error_kind(error: str) -> str | None:
    """Recognize upstream's explicit no-eligible-data signal; anything else is a failure."""
    lines = [line.strip() for line in error.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    return (
        "no-eligible-data"
        if last == f"{NO_ELIGIBLE_DATA[0]}: {NO_ELIGIBLE_DATA[1]}"
        else None
    )


def _write_sidecar(out: Path, user: int, sidecar: dict) -> None:
    path = out / f"{user}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(sidecar, indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


# ------------------------------------------------------------------- parent / readers


def read_sidecar(cache_dir, user: int) -> dict:
    path = Path(cache_dir) / f"{user}.json"
    if not path.exists():
        raise CacheError(f"no sidecar for user {user} at {path}")
    sidecar = json.loads(path.read_text())
    if sidecar.get("user") != int(user):
        raise CacheError(
            f"{path}: sidecar is for user {sidecar.get('user')}, not {user}"
        )
    if sidecar.get("identity") != identity_digest(sidecar.get("identity_fields", {})):
        raise CacheError(
            f"{path}: identity digest does not match its own identity fields"
        )
    if sidecar.get("status") == "ok" and sidecar.get("cache") != f"{user}.npz":
        raise CacheError(f"{path}: unexpected cache filename")
    return sidecar


def load_user(cache_dir, user: int) -> dict:
    """Validate sidecar + npz hash + array contract and return the loaded array dict."""
    import numpy as np

    cache_dir = Path(cache_dir)
    sidecar = read_sidecar(cache_dir, user)
    if sidecar.get("status") == "excluded":
        raise CacheError(
            f"user {user} was excluded ({sidecar.get('excluded_reason')}): "
            f"{(sidecar.get('error') or '').strip().splitlines()[-1]}"
        )
    if sidecar.get("status") != "ok" or not sidecar.get("cache"):
        raise CacheError(
            f"{cache_dir / f'{user}.json'}: sidecar status {sidecar.get('status')!r}"
        )

    npz_path = cache_dir / sidecar["cache"]
    if not npz_path.exists():
        raise CacheError(f"cache file missing: {npz_path}")
    digest = _sha256_file(npz_path)
    if digest != sidecar.get("cache_sha256"):
        raise CacheError(
            f"{npz_path}: sha256 {digest} != sidecar cache_sha256 {sidecar.get('cache_sha256')}"
        )

    with np.load(npz_path) as data:
        names = set(data.files)
        if names != set(CACHE_SPEC):
            raise CacheError(
                f"{npz_path}: array names {sorted(names)} != contract {sorted(CACHE_SPEC)}"
            )
        arrays = {name: data[name] for name in CACHE_SPEC}

    lengths = {name: len(arr) for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise CacheError(f"{npz_path}: array lengths differ: {lengths}")
    for name, dtype in CACHE_SPEC.items():
        if arrays[name].ndim != 1:
            raise CacheError(f"{npz_path}: {name} must be one-dimensional")
        if arrays[name].dtype != np.dtype(dtype):
            raise CacheError(
                f"{npz_path}: {name} dtype {arrays[name].dtype} != {dtype}"
            )
    if sidecar.get("rows") != len(arrays["y"]):
        raise CacheError(
            f"{npz_path}: {len(arrays['y'])} rows != sidecar rows {sidecar.get('rows')}"
        )

    if not (np.diff(arrays["review_th"]) > 0).all():
        raise CacheError(f"{npz_path}: review_th is not strictly increasing")
    fold = arrays["fold"].astype(np.int64)
    if (np.diff(fold) < 0).any():
        raise CacheError(
            f"{npz_path}: folds are not non-decreasing (fold-major order broken)"
        )
    if set(np.unique(fold).tolist()) != set(range(N_SPLITS)):
        raise CacheError(f"{npz_path}: fold values {np.unique(fold).tolist()}")
    if not (np.isin(arrays["y"], (0.0, 1.0)).all()):
        raise CacheError(f"{npz_path}: y is not binary")
    probability = arrays["p_fsrs"]
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise CacheError(f"{npz_path}: invalid upstream probabilities")
    if (np.diff(arrays["day_offset"]) < 0).any():
        raise CacheError(f"{npz_path}: day offsets are not chronological")
    stats = sidecar.get("stats", {})
    if stats.get("user") != user or stats.get("size") != len(arrays["y"]):
        raise CacheError(f"{npz_path}: upstream stats identity/coverage mismatch")
    return arrays


def source_identity(upstream, flags: list[str] | None = None) -> dict:
    """Public provenance helper for run.py's manifest: commit, source sha256, flags."""
    return {**upstream_source(Path(upstream)), "flags": list(flags or OFFICIAL_FLAGS)}


def _child_python(upstream: Path, override: str | None) -> str:
    if override:
        return override
    candidate = Path(upstream) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


@lru_cache(maxsize=4)
def _runtime_versions(executable: str) -> dict:
    if Path(executable).absolute() == Path(sys.executable).absolute():
        return env_versions()
    code = (
        "import json,runpy,sys; "
        "print(json.dumps(runpy.run_path(sys.argv[1])['env_versions']()))"
    )
    output = subprocess.check_output(
        [executable, "-c", code, str(Path(__file__).resolve())],
        text=True,
        env=_child_env(),
    )
    return json.loads(output)


def _child_env() -> dict:
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="-1",
    )
    return env


def _resolve(upstream, data, out) -> tuple[Path, Path, Path]:
    upstream, data, out = (
        Path(upstream).resolve(),
        Path(data).resolve(),
        Path(out).resolve(),
    )
    if not (upstream / "script.py").is_file():
        raise RuntimeError(f"no upstream script.py under {upstream}")
    if not (data / "revlogs").is_dir():
        raise RuntimeError(f"no revlogs under {data}")
    out.mkdir(parents=True, exist_ok=True)
    return upstream, data, out


def run_user(
    upstream,
    data,
    out,
    user: int,
    dataset_revision: str | None = None,
    python: str | None = None,
    force: bool = False,
) -> dict:
    """Run upstream for one user in a fresh process; write cache + sidecar; return sidecar.

    Returns the sidecar dict, plus an output-only ``resumed: True`` marker when the user was
    skipped because a matching keyed cache already existed.  Raises CacheError when an
    existing cache's identity differs (unless ``force``) and RuntimeError when the user
    fails for an unexpected reason.
    """
    upstream, data, out = _resolve(upstream, data, out)
    user = int(user)
    files = data_files(data, user)
    executable = _child_python(upstream, python)
    fields = identity_fields(
        upstream,
        OFFICIAL_FLAGS,
        dataset_revision,
        user,
        files,
        runtime=_runtime_versions(executable),
    )
    expected = identity_digest(fields)

    if not force:
        existing = _existing_state(out, user, fields, expected)
        if existing is not None:
            return {**existing, "resumed": True}

    cmd = [
        executable,
        str(Path(__file__).resolve()),
        "--upstream",
        str(upstream),
        "--data",
        str(data),
        "--out",
        str(out),
        "--user",
        str(user),
    ]
    if dataset_revision is not None:
        cmd += ["--dataset-revision", dataset_revision]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=_child_env(), cwd=str(out), check=False
    )
    if proc.returncode not in (0, 3):
        raise RuntimeError(
            f"user {user}: runner child exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    sidecar = read_sidecar(out, user)
    if sidecar["identity"] != expected:
        raise CacheError(f"user {user}: sidecar identity changed during the run")
    if sidecar["status"] == "ok" and not (out / sidecar["cache"]).exists():
        raise RuntimeError(f"user {user}: child reported ok but wrote no cache")
    return sidecar


def _existing_state(out: Path, user: int, fields: dict, expected: str) -> dict | None:
    """Resume state for a finished user, or None to (re)run.  Refuses mismatches."""
    if not (out / f"{user}.json").exists():
        if (out / f"{user}.npz").exists():
            raise CacheError(
                f"{out / f'{user}.npz'}: cache without a sidecar (legacy/unkeyed) - "
                "refusing to reuse; delete it or pass --force"
            )
        return None
    sidecar = read_sidecar(out, user)
    if sidecar["identity"] == expected:
        if sidecar["status"] == "ok":
            path = out / sidecar["cache"]
            if not path.exists() or _sha256_file(path) != sidecar["cache_sha256"]:
                raise CacheError(f"{path}: cached file does not match its sidecar hash")
        return sidecar
    raise CacheError(
        f"user {user}: cache identity mismatch on resume\n"
        + "\n".join(
            f"  {key}: cached={v!r} expected={fields.get(key)!r}"
            for key, v in sidecar["identity_fields"].items()
            if v != fields.get(key)
        )
        + "\n  (delete the sidecar or pass --force to re-run)"
    )


def user_list(data: Path, users_arg: str | None, all_users: bool) -> list[int]:
    if all_users and users_arg:
        raise RuntimeError("--users and --all are mutually exclusive")
    if all_users:
        ids = [
            int(p.name.split("=", 1)[1])
            for p in (data / "revlogs").glob("user_id=*")
            if p.is_dir()
        ]
        return sorted(ids)
    if not users_arg:
        raise RuntimeError("pass --users JSON or --all")
    payload = json.loads(Path(users_arg).read_text())
    if isinstance(payload, dict):
        payload = payload.get("user_ids")
    if not isinstance(payload, list) or not all(
        isinstance(u, (int, str)) for u in payload
    ):
        raise RuntimeError(
            "--users must be a JSON list of ids or an object with user_ids"
        )
    ids = [int(u) for u in payload]
    if len(set(ids)) != len(ids):
        raise RuntimeError("--users contains duplicate ids")
    return sorted(ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinned-upstream baseline runner")
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    parser.add_argument(
        "--data", required=False, help="dataset root containing revlogs/ (and cards/)"
    )
    parser.add_argument("--out", required=False, help="cache directory")
    parser.add_argument(
        "--users", default=None, help="JSON list of ids or {'user_ids': [...]}"
    )
    parser.add_argument(
        "--all", action="store_true", help="scan <data>/revlogs for users"
    )
    parser.add_argument(
        "--user",
        type=int,
        default=None,
        help="child mode: run exactly this user in-process",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="concurrent users (1 or 2)"
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="caller-supplied dataset revision, recorded in the identity",
    )
    parser.add_argument("--python", default=None, help="child interpreter override")
    parser.add_argument(
        "--force", action="store_true", help="re-run even if a keyed cache exists"
    )
    args = parser.parse_args(argv)

    if args.user is not None:
        upstream, data, out = _resolve(args.upstream, args.data, args.out)
        sidecar = _capture_user(upstream, data, out, args.user, args.dataset_revision)
        if sidecar["status"] == "ok":
            return 0
        if sidecar["status"] == "excluded":
            return 3
        print(sidecar["error"] or "user failed", file=sys.stderr)
        return 1

    if not args.data or not args.out:
        parser.error("--data and --out are required")
    if args.workers not in (1, 2):
        parser.error("--workers must be 1 or 2")
    upstream, data, out = _resolve(args.upstream, args.data, args.out)
    users = user_list(data, args.users, args.all)
    src = upstream_source(upstream)
    print(
        f"upstream {src['commit']} source_sha256={src['sha256'][:12]} "
        f"files={src['n_present']}/{src['n_tracked']} users={len(users)} "
        f"workers={args.workers} out={out}",
        flush=True,
    )
    print(
        f"flags {' '.join(OFFICIAL_FLAGS)} data={data} revision={args.dataset_revision}",
        flush=True,
    )

    t0 = time.monotonic()
    counts = {"ok": 0, "resumed": 0, "excluded": 0, "failed": 0, "refused": 0}
    failures: list[str] = []

    def work(user: int) -> tuple[int, str, str]:
        try:
            sidecar = run_user(
                upstream,
                data,
                out,
                user,
                args.dataset_revision,
                args.python,
                args.force,
            )
        except CacheError as exc:
            return user, "refused", str(exc)
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            return user, "failed", str(exc)
        if sidecar["status"] == "ok":
            if sidecar.get("resumed"):
                return user, "resumed", f"{sidecar['rows']} rows"
            return user, "ok", f"{sidecar['rows']} rows {sidecar['elapsed_seconds']}s"
        if sidecar["status"] == "excluded":
            return user, "excluded", str(sidecar.get("excluded_reason"))
        return user, "failed", str(sidecar.get("error"))

    total, done = len(users), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, user) for user in users]
        for future in as_completed(futures):
            user, state, detail = future.result()
            counts[state] += 1
            done += 1
            if state in ("failed", "refused"):
                failures.append(f"user {user}: {detail}")
            print(f"[{done}/{total}] user={user} {state} {detail}", flush=True)

    elapsed = round(time.monotonic() - t0, 1)
    print(
        f"done in {elapsed}s: ok={counts['ok']} resumed={counts['resumed']} "
        f"excluded={counts['excluded']} refused={counts['refused']} failed={counts['failed']}",
        flush=True,
    )
    for message in failures:
        print(f"FAILED {message.splitlines()[0]}", flush=True)
    return 1 if (counts["failed"] or counts["refused"]) else 0


if __name__ == "__main__":
    sys.exit(main())
