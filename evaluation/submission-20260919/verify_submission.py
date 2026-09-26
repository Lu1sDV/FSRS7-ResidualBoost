#!/usr/bin/env python
"""Verify distributed source/model/result integrity without raw review data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_MODELS = ("FSRS-7", "B", "FSRS7-ResidualBoost")

sys.path.insert(0, str(HERE))
import baseline
import nested
from run import digest
from test_submission import synthetic


def _exact_user_ids(values, label):
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a JSON list")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} contains a non-integer user id: {value!r}")
        result.append(value)
    if result != sorted(set(result)):
        raise ValueError(f"{label} must be sorted and contain unique user ids")
    return result


def _reference_projection(path: Path):
    """Read the distributed user/size projection of the upstream reference result."""

    users = []
    sizes = []
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid upstream reference JSON at line {line_number}"
                ) from exc
            if set(row) != {"user", "size"}:
                raise ValueError(
                    f"upstream reference line {line_number} is not a user/size projection"
                )
            user, size = row["user"], row["size"]
            if isinstance(user, bool) or not isinstance(user, int):
                raise ValueError(
                    f"upstream reference line {line_number} has non-integer user"
                )
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ValueError(
                    f"upstream reference line {line_number} has invalid size"
                )
            if user in seen:
                raise ValueError(f"duplicate upstream reference user {user}")
            seen.add(user)
            users.append(user)
            sizes.append(size)
    if users != sorted(users):
        raise ValueError("upstream reference users are not sorted")
    return users, sizes


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def validate_frozen_contract(root: Path, manifest: dict, report: dict, qa_only: bool):
    """Validate claims that do not require raw reviews or upstream source."""

    here = root / "evaluation/submission-20260919"
    protocol = json.loads((here / "protocol.json").read_text())
    users = _exact_user_ids(manifest.get("users"), "manifest users")
    deferred = _exact_user_ids(
        manifest.get("deferred_users") or [],
        "manifest deferred_users",
    )

    if set(report.get("models", {})) != set(EXPECTED_MODELS):
        raise ValueError(
            "result model set differs from the frozen submission: "
            f"{sorted(report.get('models', {}))}"
        )
    if report.get("users") != len(users):
        raise ValueError("result user count differs from frozen manifest")
    if manifest.get("upstream_revision") != protocol.get("upstream_revision"):
        raise ValueError("manifest upstream revision differs from protocol.json")
    if manifest.get("dataset_revision") != protocol.get("dataset_revision"):
        raise ValueError("manifest dataset revision differs from protocol.json")
    upstream_source = manifest.get("upstream_source")
    if not isinstance(upstream_source, dict):
        raise ValueError("manifest upstream_source is missing")
    if upstream_source.get("commit") != protocol.get("upstream_revision"):
        raise ValueError("recorded upstream source commit differs from protocol.json")

    protocol_sha = digest(here / "protocol.json")
    if manifest.get("protocol_sha256") != protocol_sha:
        raise ValueError("manifest protocol_sha256 does not match protocol.json")

    reference_path = here / "upstream-reference.jsonl"
    reference_sha = digest(reference_path)
    if manifest.get("reference_sha256") != reference_sha:
        raise ValueError("manifest reference_sha256 does not match upstream-reference.jsonl")
    reference_users, reference_sizes = _reference_projection(reference_path)

    # protocol.coverage_reference.upstream_sha256 is the hash of upstream's
    # original full result JSONL. upstream-reference.jsonl is intentionally a
    # data-minimized user/size projection, so the two hashes must not be equated.
    coverage_reference = protocol.get("coverage_reference")
    if not isinstance(coverage_reference, dict):
        raise ValueError("protocol coverage_reference is missing")
    if not _is_sha256(coverage_reference.get("upstream_sha256")):
        raise ValueError("protocol upstream result hash is not SHA-256")
    if coverage_reference.get("upstream_path") != (
        "result/FSRS-7-short-secs-recency-equalize_test_with_non_secs.jsonl"
    ):
        raise ValueError("protocol upstream result path differs from the frozen reference")
    if coverage_reference.get("projection") != (
        "user and size only; no reference metrics or fitted parameters are evaluation inputs"
    ):
        raise ValueError("protocol coverage-reference projection contract differs")

    verification = json.loads((here / "baseline-verification.json").read_text())
    if verification.get("upstream_commit") != protocol.get("upstream_revision"):
        raise ValueError("baseline verification used a different upstream commit")
    if verification.get("capture_sha256") != digest(here / "baseline.py"):
        raise ValueError("baseline verification capture hash differs from baseline.py")
    if verification.get("max_probability_difference") != 0.0:
        raise ValueError("baseline verification is not probability-exact")
    if verification.get("stock_metrics_parameters_and_coverage_equal") is not True:
        raise ValueError("baseline verification did not reproduce stock metrics/parameters")
    if verification.get("stock_fold_membership_equal") is not True:
        raise ValueError("baseline verification did not reproduce stock fold membership")

    if qa_only:
        print("QA-only archive: frozen-population reconciliation skipped")
        return protocol, users, deferred

    frozen_payload = json.loads((here / "users.json").read_text())
    frozen_ids = _exact_user_ids(
        frozen_payload["user_ids"] if isinstance(frozen_payload, dict) else frozen_payload,
        "frozen users",
    )
    if manifest.get("frozen_users") != len(frozen_ids):
        raise ValueError("manifest frozen user count differs from the frozen population")
    if set(users) & set(deferred) or sorted(users + deferred) != frozen_ids:
        raise ValueError("evaluated plus deferred users do not equal the frozen population")
    if reference_users != frozen_ids:
        raise ValueError("upstream-reference user IDs differ from the frozen population")
    if sum(reference_sizes) != protocol.get("expected_scored_reviews"):
        raise ValueError(
            "upstream-reference scored-review coverage differs from protocol"
        )
    if deferred and not (root / "COVERAGE-GAP.txt").is_file():
        raise ValueError("deferred coverage must ship a visible COVERAGE-GAP.txt")

    expected_rounds = protocol.get("booster", {}).get("num_boost_round")
    if manifest.get("rounds") != expected_rounds:
        raise ValueError(
            f"run rounds {manifest.get('rounds')} != frozen protocol {expected_rounds}"
        )
    if not deferred and report.get("reviews") != protocol.get("expected_scored_reviews"):
        raise ValueError(
            "full-coverage report review count differs from protocol expected_scored_reviews"
        )
    return protocol, users, deferred


def verify_upstream_checkout(upstream: Path, manifest: dict) -> None:
    """Verify the separately obtained sparse upstream checkout used for reproduction."""

    upstream = upstream.resolve()
    if not (upstream / ".git").exists():
        raise ValueError(f"upstream checkout is not a git repository: {upstream}")

    expected = manifest.get("upstream_source")
    if not isinstance(expected, dict):
        raise ValueError("manifest upstream_source is missing")
    actual = baseline.source_identity(upstream, expected.get("flags"))
    if actual != expected:
        keys = sorted(set(actual) | set(expected))
        mismatch = {
            key: {"recorded": expected.get(key), "actual": actual.get(key)}
            for key in keys
            if actual.get(key) != expected.get(key)
        }
        raise ValueError(f"upstream source identity differs from recorded run: {mismatch}")

    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(upstream),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if status.strip():
        raise ValueError(
            "upstream checkout has tracked modifications; reproduction must use "
            "the exact recorded sparse source tree"
        )
    print(
        "upstream: exact recorded sparse checkout "
        f"{expected['commit']} / {expected['sha256']} OK"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--upstream",
        type=Path,
        help=(
            "optional separately cloned srs-benchmark checkout; when supplied, "
            "verify its commit, sparse file set, source bytes and clean tracked state "
            "against the recorded full-run manifest"
        ),
    )
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
    report = json.loads((result_dir / "results.json").read_text())
    qa_only = (root / "SMOKE-ONLY.txt").is_file()
    protocol, users, deferred = validate_frozen_contract(
        root,
        manifest,
        report,
        qa_only,
    )

    if args.upstream is not None:
        verify_upstream_checkout(args.upstream, manifest)

    for relative, expected in manifest["source_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"training source identity mismatch: {relative}")

    row_counts = None
    results_by_model = {}
    for model in EXPECTED_MODELS:
        expected_metrics = report["models"][model]
        result_path = result_dir / f"result-{model}.jsonl"
        records = [json.loads(line) for line in result_path.read_text().splitlines()]
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
    if len(split) != protocol.get("outer_folds"):
        raise ValueError(
            f"stored split count {len(split)} != protocol outer_folds "
            f"{protocol.get('outer_folds')}"
        )
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
        "PASS: frozen contract, source, models, split lineage, result coverage and "
        "synthetic inference agree"
    )
    print(
        "This is a nested refit on a previously studied public benchmark, not a "
        "prospective new-data claim."
    )


if __name__ == "__main__":
    main()
