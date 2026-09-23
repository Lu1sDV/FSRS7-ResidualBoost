"""Focused tests for baseline.py's cache contract: identity, refusal, chronology.

The loader is the only accessor the nested stages use, so a cache that is unkeyed,
truncated, mis-shaped or out of order must raise CacheError instead of being read.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline


def _arrays(n=10):
    t = np.arange(n, dtype=np.int64)
    return {
        "y": (t % 2).astype(np.float64),
        "p_fsrs": (0.1 + 0.05 * t).astype(np.float64),
        "review_th": t + 1,
        "fold": (t // 2).astype(np.int8),
        "i": (t % 3 + 1).astype(np.int64),
        "elapsed_days": (t % 4 + 1).astype(np.float64),
        "lapse": (t // 4).astype(np.int64),
        "nth_today": (t % 2 + 1).astype(np.int64),
        "day_offset": (t // 2).astype(np.int64),
        "card_id": (t % 3).astype(np.int64),
        "deck_id": (t % 3).astype(np.int64),
        "note_id": (t % 3).astype(np.int64),
        "rating": (t % 4 + 1).astype(np.float64),
        "elapsed_seconds": (t % 4 + 1) * 86400.0,
        "duration": (t % 6 + 1) * 900.0,
    }


def _write_cache(tmp_path, user=7, arrays=None):
    arrays = arrays or _arrays()
    npz = tmp_path / f"{user}.npz"
    with npz.open("wb") as f:
        np.savez(f, allow_pickle=False, **arrays)
    fields = {
        "upstream_commit": "bd9110f",
        "upstream_source_sha256": "0" * 64,
        "upstream_source_files": 64,
        "flags": baseline.OFFICIAL_FLAGS,
        "dataset_revision": "rev-a",
        "user": user,
        "data_files": {
            "revlogs": {"sha256": "1" * 64, "bytes": 10, "files": ["data.parquet"]},
            "cards": None,
        },
    }
    sidecar = {
        "user": user,
        "status": "ok",
        "rows": len(arrays["y"]),
        "cache": npz.name,
        "cache_sha256": baseline._sha256_file(npz),
        "stats": {"metrics": {"LogLoss": 0.5}, "user": user, "size": len(arrays["y"])},
        "identity": baseline.identity_digest(fields),
        "identity_fields": fields,
    }
    (tmp_path / f"{user}.json").write_text(json.dumps(sidecar))
    return sidecar


def test_load_user_roundtrip(tmp_path):
    _write_cache(tmp_path)
    arrays = baseline.load_user(tmp_path, 7)
    assert set(arrays) == set(baseline.CACHE_SPEC)
    for name, dtype in baseline.CACHE_SPEC.items():
        assert arrays[name].dtype == np.dtype(dtype)
        assert len(arrays[name]) == 10
    np.testing.assert_array_equal(arrays["review_th"], np.arange(1, 11))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda a: {**a, "review_th": a["review_th"][::-1].copy()},
            "strictly increasing",
        ),
        (lambda a: {**a, "fold": np.zeros_like(a["fold"])}, "fold values"),
        (lambda a: {**a, "i": a["i"].astype(np.int32)}, "dtype"),
        (lambda a: {**a, "y": np.full_like(a["y"], 2.0)}, "not binary"),
        (lambda a: {**a, "duration": a["duration"][:-1].copy()}, "lengths differ"),
        (lambda a: {k: v for k, v in a.items() if k != "lapse"}, "array names"),
    ],
)
def test_load_user_refuses_bad_cache(tmp_path, mutate, message):
    arrays = _arrays()
    _write_cache(tmp_path)
    bad = mutate(arrays)
    npz = tmp_path / "7.npz"
    with npz.open("wb") as f:
        np.savez(f, allow_pickle=False, **bad)
    sidecar = json.loads((tmp_path / "7.json").read_text())
    sidecar["cache_sha256"] = baseline._sha256_file(
        npz
    )  # isolate the shape/order check
    (tmp_path / "7.json").write_text(json.dumps(sidecar))
    with pytest.raises(baseline.CacheError, match=message):
        baseline.load_user(tmp_path, 7)


def test_load_user_refuses_hash_mismatch_and_legacy_cache(tmp_path):
    _write_cache(tmp_path)
    (tmp_path / "7.npz").write_bytes((tmp_path / "7.npz").read_bytes() + b"x")
    with pytest.raises(baseline.CacheError, match="sha256"):
        baseline.load_user(tmp_path, 7)

    (tmp_path / "7.json").unlink()
    with pytest.raises(baseline.CacheError, match="sidecar"):
        baseline.load_user(tmp_path, 7)
    with pytest.raises(baseline.CacheError, match="sidecar"):
        baseline._existing_state(tmp_path, 7, {}, "unused")


def test_identity_digest_is_content_keyed(tmp_path):
    files = {
        "revlogs": {"sha256": "a" * 64, "bytes": 1, "files": ["data.parquet"]},
        "cards": None,
    }
    base = baseline.identity_fields(
        baseline.DEFAULT_UPSTREAM, baseline.OFFICIAL_FLAGS, "rev-a", 7, files
    )
    other_rev = baseline.identity_fields(
        baseline.DEFAULT_UPSTREAM, baseline.OFFICIAL_FLAGS, "rev-b", 7, files
    )
    other_data = baseline.identity_fields(
        baseline.DEFAULT_UPSTREAM,
        baseline.OFFICIAL_FLAGS,
        "rev-a",
        7,
        {
            **files,
            "revlogs": {"sha256": "b" * 64, "bytes": 1, "files": ["data.parquet"]},
        },
    )
    digests = {baseline.identity_digest(f) for f in (base, other_rev, other_data)}
    assert len(digests) == 3
    assert baseline.identity_digest(base) == baseline.identity_digest(dict(base))


def test_only_upstream_no_eligible_data_is_an_exclusion():
    signal = f"{baseline.NO_ELIGIBLE_DATA[0]}: {baseline.NO_ELIGIBLE_DATA[1]}"
    assert (
        baseline._error_kind(f"User 4371:\nTraceback\n{signal}\n") == "no-eligible-data"
    )
    assert baseline._error_kind("ValueError: something else") is None
    assert baseline._error_kind(f"User 1:\nTraceback\n{signal} (extra)\n") is None
    assert baseline._error_kind("") is None


def test_user_list_accepts_list_or_mapping_and_rejects_junk(tmp_path):
    (tmp_path / "list.json").write_text("[3, 1, 2]")
    (tmp_path / "map.json").write_text('{"user_ids": ["4", 5]}')
    (tmp_path / "dups.json").write_text("[1, 1]")
    assert baseline.user_list(tmp_path, str(tmp_path / "list.json"), False) == [1, 2, 3]
    assert baseline.user_list(tmp_path, str(tmp_path / "map.json"), False) == [4, 5]
    with pytest.raises(RuntimeError, match="duplicate"):
        baseline.user_list(tmp_path, str(tmp_path / "dups.json"), False)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        baseline.user_list(tmp_path, str(tmp_path / "list.json"), True)
    with pytest.raises(RuntimeError, match="pass --users"):
        baseline.user_list(tmp_path, None, False)


def test_identity_changes_with_runtime_versions(monkeypatch):
    files = {"revlogs": None, "cards": None}
    monkeypatch.setattr(
        baseline, "env_versions", lambda: {"python": "3.14", "torch": "A"}
    )
    first = baseline.identity_fields(
        baseline.DEFAULT_UPSTREAM, baseline.OFFICIAL_FLAGS, "revision", 7, files
    )
    monkeypatch.setattr(
        baseline, "env_versions", lambda: {"python": "3.14", "torch": "B"}
    )
    second = baseline.identity_fields(
        baseline.DEFAULT_UPSTREAM, baseline.OFFICIAL_FLAGS, "revision", 7, files
    )
    assert baseline.identity_digest(first) != baseline.identity_digest(second)


def test_identity_changes_with_capture_wrapper(monkeypatch, tmp_path):
    wrapper = tmp_path / "baseline.py"
    wrapper.write_text("first capture implementation")
    monkeypatch.setattr(baseline, "__file__", str(wrapper))
    first = baseline.identity_fields(
        baseline.DEFAULT_UPSTREAM, baseline.OFFICIAL_FLAGS, "revision", 7, {}
    )
    wrapper.write_text("changed capture implementation")
    second = baseline.identity_fields(
        baseline.DEFAULT_UPSTREAM, baseline.OFFICIAL_FLAGS, "revision", 7, {}
    )
    assert baseline.identity_digest(first) != baseline.identity_digest(second)


@pytest.mark.parametrize("value", [float("nan"), 1.1])
def test_loader_rejects_invalid_probability_even_with_valid_hash(tmp_path, value):
    arrays = _arrays()
    arrays["p_fsrs"][0] = value
    _write_cache(tmp_path, arrays=arrays)
    with pytest.raises(baseline.CacheError):
        baseline.load_user(tmp_path, 7)
