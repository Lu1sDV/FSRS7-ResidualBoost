#!/usr/bin/env python
"""Verify distributed source/model/result integrity without raw review data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import nested
from run import digest
from test_submission import synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    checksums = root / "SHA256SUMS"
    listed = {}
    for line in checksums.read_text().splitlines():
        value, relative = line.split("  ", 1)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in listed:
            raise ValueError(f"unsafe or duplicate checksum path: {relative}")
        listed[relative] = value
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlinks are not permitted in the submission archive")
    actual = {str(p.relative_to(root)) for p in files if p.name != "SHA256SUMS"}
    if actual != set(listed):
        raise ValueError("archive file coverage differs from SHA256SUMS")
    for relative, expected in listed.items():
        if digest(root / relative) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    if not (root / "scripts/fullrun-chain-vps.sh").is_file():
        raise ValueError("executed full-run procedure is missing from the archive")
    print(f"checksums: {len(listed)} files OK")
    here = root / "evaluation/submission-20260919"
    result_dir = here / "run"
    manifest = json.loads((result_dir / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"training source identity mismatch: {relative}")
    users = manifest["users"]
    if len(users) != len(set(users)):
        raise ValueError("duplicate evaluation users")
    deferred = manifest.get("deferred_users") or []
    qa_only = (root / "SMOKE-ONLY.txt").is_file()
    if qa_only:
        # QA archives evaluate an arbitrary subset and never claim frozen coverage.
        print("QA-only archive: frozen-population reconciliation skipped")
    else:
        frozen = json.loads(
            (root / "evaluation/submission-20260919/users.json").read_text()
        )
        frozen_count = len(frozen["user_ids"] if isinstance(frozen, dict) else frozen)
        if manifest.get("frozen_users") != frozen_count:
            raise ValueError(
                "manifest frozen user count differs from the frozen population"
            )
        if len(users) + len(deferred) != frozen_count or set(users) & set(deferred):
            raise ValueError("deferred users do not reconcile with frozen coverage")
        if deferred and not (root / "COVERAGE-GAP.txt").is_file():
            raise ValueError("deferred coverage must ship a visible COVERAGE-GAP.txt")
    report = json.loads((result_dir / "results.json").read_text())
    if report["users"] != len(users):
        raise ValueError("result user count differs from frozen manifest")
    row_counts = None
    results_by_model = {}
    for model, expected_metrics in report["models"].items():
        records = [
            json.loads(line)
            for line in (result_dir / f"result-{model}.jsonl").read_text().splitlines()
        ]
        results_by_model[model] = records
        if [r["user"] for r in records] != users:
            raise ValueError(f"{model}: record IDs differ from frozen coverage")
        sizes = [r["size"] for r in records]
        if sum(sizes) != report["reviews"] or (
            row_counts is not None and sizes != row_counts
        ):
            raise ValueError(f"{model}: inconsistent scored row counts")
        row_counts = sizes
        for metric in ("LogLoss", "RMSE(bins)", "AUC"):
            values = [
                r["metrics"][metric]
                for r in records
                if r["metrics"][metric] is not None
            ]
            if metric != "AUC" and len(values) != len(users):
                raise ValueError(f"{model}: null {metric}")
            if (
                not np.isfinite(values).all()
                or abs(float(np.mean(values)) - expected_metrics[metric]["mean"])
                > 1e-12
            ):
                raise ValueError(f"{model}: saved {metric} mean does not reconcile")
            if len(values) != expected_metrics[metric]["n_users"]:
                raise ValueError(f"{model}: metric coverage mismatch")
        print(f"{model}: {len(records)} users / {sum(sizes)} rows OK")
    if row_counts is None:
        raise ValueError("no model results were checked")
    base_provenance = [
        json.loads(line)
        for line in (result_dir / "base-provenance.jsonl").read_text().splitlines()
    ]
    score_provenance = [
        json.loads(line)
        for line in (result_dir / "score-provenance.jsonl").read_text().splitlines()
    ]
    if [row["user"] for row in base_provenance] != users:
        raise ValueError("base provenance IDs differ from frozen coverage")
    if [row["identity"]["user"] for row in score_provenance] != users:
        raise ValueError("score provenance IDs differ from frozen coverage")
    for position, user in enumerate(users):
        base = base_provenance[position]
        score = score_provenance[position]
        identity_fields = base["identity_fields"]
        if (
            identity_fields is None
            or base["rows"] != row_counts[position]
            or identity_fields["user"] != user
            or base["metrics"] != results_by_model["FSRS-7"][position]["metrics"]
        ):
            raise ValueError(f"user {user}: base provenance does not reconcile")
        for model, records in results_by_model.items():
            if score["records"][model] != records[position]:
                raise ValueError(f"user {user}: score provenance does not reconcile")
    print(f"provenance: {len(users)} base and score records reconcile")
    base_by_user = {row["user"]: row for row in base_provenance}
    score_by_user = {row["identity"]["user"]: row for row in score_provenance}
    split = json.loads((result_dir / "splits.json").read_text())
    held_all = []
    for index, fold in enumerate(split):
        train, val, held = map(set, (fold["train"], fold["val"], fold["held"]))
        if (
            train & val
            or train & held
            or val & held
            or train | val | held != set(users)
        ):
            raise ValueError(f"fold {index}: invalid user partition")
        held_all.extend(held)
        fit = json.loads((result_dir / f"fold-{index}/fit.json").read_text())
        if fit["split"] != fold:
            raise ValueError(f"fold {index}: fitted split differs")
        if (
            fit["model"] != "model.json"
            or fit["calibration_json"] != "calibration.json"
            or fit["lineage_json"] != "lineage.json"
        ):
            raise ValueError(f"fold {index}: fitted artifact paths are not portable")
        for user in held:
            identity = score_by_user[user]["identity"]
            if (
                identity["fold"] != index
                or identity["fit_sha256"] != fit["original_fit_sha256"]
                or identity["base_sha256"] != base_by_user[user]["cache_sha256"]
            ):
                raise ValueError(f"user {user}: held-score dependency mismatch")
        calibrations = fit["calibration_fits"]
        inner = calibrations[:-1]
        validation = calibrations[-1]
        applied_train = []
        for calibration in inner:
            fitting, applying = map(
                set, (calibration["fit_users"], calibration["apply_users"])
            )
            if fitting != train - applying or not applying <= train:
                raise ValueError(f"fold {index}: invalid inner calibration cross-fit")
            applied_train.extend(applying)
        if sorted(applied_train) != sorted(train):
            raise ValueError(
                f"fold {index}: training users not cross-fitted exactly once"
            )
        if (
            validation["fold"] != "val"
            or set(validation["fit_users"]) != train
            or set(validation["apply_users"]) != val
            or validation["beta"] != fit["beta"]
        ):
            raise ValueError(f"fold {index}: invalid validation/held calibration fit")
        model = result_dir / f"fold-{index}" / Path(fit["model"]).name
        if digest(model) != fit["model_sha256"]:
            raise ValueError(f"fold {index}: booster identity mismatch")
        booster = xgb.Booster(model_file=str(model))
        sample = synthetic(1)
        offline = nested.predict_user(sample, fit["beta"], booster, threads=1)
        sys.path.insert(0, str(root / "evaluation/tabular-probe-20260917"))
        from serve import StreamScorer

        stream = StreamScorer(model)
        stream.begin_user(fit["beta"])
        online = [
            stream.observe(
                y=float(sample["y"][t]),
                p_fsrs=float(sample["p_fsrs"][t]),
                i=int(sample["i"][t]),
                elapsed_days=float(sample["elapsed_days"][t]),
                lapse=int(sample["lapse"][t]),
                nth_today=int(sample["nth_today"][t]),
                day=float(sample["day_offset"][t]),
                card_id=int(sample["card_id"][t]),
                deck_id=int(sample["deck_id"][t]),
                note_id=int(sample["note_id"][t]),
                rating=float(sample["rating"][t]),
                elapsed_seconds=float(sample["elapsed_seconds"][t]),
                duration=float(sample["duration"][t]),
            )
            for t in range(len(sample["y"]))
        ]
        delta = float(np.max(np.abs(offline - online)))
        if delta > 1e-6:
            raise ValueError(f"fold {index}: offline/online inference mismatch {delta}")
        print(
            f"fold {index}: isolated calibration lineage and synthetic inference parity {delta:.2e} OK"
        )
    if sorted(held_all) != users:
        raise ValueError("not every evaluation user was held exactly once")
    print(
        "PASS: source, models, split lineage, result coverage and synthetic inference agree"
    )
    print(
        "This is a nested refit on a previously studied public benchmark, not a prospective new-data claim."
    )


if __name__ == "__main__":
    main()
