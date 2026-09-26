"""Regression tests for post-evaluation report and streaming hardening."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROBE = ROOT / "evaluation/tabular-probe-20260917"
for path in (HERE, PROBE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import baseline
import run_hardened
from serve_hardened import StreamScorer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_report_run(tmp_path: Path, user: int = 1) -> Path:
    out = tmp_path / "run"
    cache = out / "base"
    scores = out / "scores"
    fold_dir = out / "fold-0"
    for path in (cache, scores, fold_dir):
        path.mkdir(parents=True)

    reference = {
        row["user"]: row["size"]
        for row in map(
            json.loads,
            (HERE / "upstream-reference.jsonl").read_text().splitlines(),
        )
    }
    size = reference[user]
    fold = {"train": [], "val": [], "held": [user]}
    (out / "manifest.json").write_text(json.dumps({"users": [user]}))
    (out / "splits.json").write_text(json.dumps([fold]))

    fields = {"user": user}
    cache_sha = "a" * 64
    sidecar = {
        "user": user,
        "status": "ok",
        "rows": size,
        "cache": f"{user}.npz",
        "cache_sha256": cache_sha,
        "identity_fields": fields,
        "identity": baseline.identity_digest(fields),
    }
    (cache / f"{user}.json").write_text(json.dumps(sidecar))

    model = fold_dir / "model.json"
    model.write_text("test model bytes")
    empty_dependency = hashlib.sha256(
        json.dumps({}, sort_keys=True).encode()
    ).hexdigest()
    fit = {
        "split": fold,
        "base_dependency_sha256": empty_dependency,
        "model": "model.json",
        "model_sha256": _sha(model),
    }
    fit_path = fold_dir / "fit.json"
    fit_path.write_text(json.dumps(fit))

    records = {
        model_name: {"user": user, "size": size, "metrics": {}}
        for model_name in run_hardened.MODELS
    }
    score = {
        "identity": {
            "user": user,
            "fold": 0,
            "base_sha256": cache_sha,
            "fit_sha256": _sha(fit_path),
        },
        "records": records,
    }
    (scores / f"{user}.json").write_text(json.dumps(score))
    return out


def test_report_validation_rejects_stale_score_fit(tmp_path):
    out = _minimal_report_run(tmp_path)
    assert run_hardened.validate_report_inputs(out) > 0

    path = out / "scores/1.json"
    score = json.loads(path.read_text())
    score["identity"]["fit_sha256"] = "0" * 64
    path.write_text(json.dumps(score))

    with pytest.raises(ValueError, match="stale score provenance"):
        run_hardened.validate_report_inputs(out)


def test_report_validation_rejects_wrong_per_user_coverage(tmp_path):
    out = _minimal_report_run(tmp_path)
    path = out / "scores/1.json"
    score = json.loads(path.read_text())
    score["records"]["FSRS-7"]["size"] -= 1
    path.write_text(json.dumps(score))

    with pytest.raises(ValueError, match="coverage differs"):
        run_hardened.validate_report_inputs(out)


def test_stream_group_move_keeps_other_counts_nonnegative():
    scorer = StreamScorer.__new__(StreamScorer)
    scorer.begin_user([0.0, 0.0, 0.0])

    # Reproduce the old failure boundary: a card with 64 reviews moves to a
    # previously unseen deck/note. The frozen scorer divided by zero here.
    for t in range(64):
        scorer.t = t
        scorer._advance(
            card=1,
            deck=10,
            note=100,
            day=float(t),
            rB=0.1,
            elapsed=1.0,
        )

    scorer.t = 64
    moved = scorer._deck_features(card=1, deck=20, note=200, day=64.0)
    assert moved["dk_n_other"] == 0.0
    assert moved["nt_n_other"] == 0.0
    assert moved["dk_acc_other"] == 0.0
    assert moved["nt_acc_other"] == 0.0

    scorer._advance(
        card=1,
        deck=20,
        note=200,
        day=64.0,
        rB=0.2,
        elapsed=1.0,
    )
    scorer.t = 65

    # Returning to the original group excludes only this card's history in that
    # group; it never subtracts history accumulated elsewhere.
    returned = scorer._deck_features(card=1, deck=10, note=100, day=65.0)
    assert returned["dk_n_other"] == 0.0
    assert returned["nt_n_other"] == 0.0
    assert returned["dk_cards_seen"] == 1.0
