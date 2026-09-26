#!/usr/bin/env python
"""Safer front-end for the frozen evaluation runner.

The published run.py is kept byte-for-byte for provenance. This wrapper delegates
scientific stages to it, but validates score dependencies and per-user coverage
before report aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN_RUNNER = HERE / "run.py"
MODELS = ("FSRS-7", "B", "FSRS7-ResidualBoost")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_digest(cache: Path, fold: dict) -> str:
    import baseline

    dependencies = {}
    for user in sorted(fold["train"] + fold["val"]):
        sidecar = baseline.read_sidecar(cache, user)
        dependencies[str(user)] = [sidecar["identity"], sidecar["cache_sha256"]]
    return hashlib.sha256(
        json.dumps(dependencies, sort_keys=True).encode()
    ).hexdigest()


def validate_report_inputs(out: Path) -> int:
    """Refuse stale scores and return the validated scored-review total."""

    import baseline

    out = Path(out).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    users = manifest["users"]
    if users != sorted(set(users)):
        raise ValueError("manifest users must be sorted and unique")

    split = json.loads((out / "splits.json").read_text())
    reference = {
        row["user"]: row
        for row in map(
            json.loads,
            (HERE / "upstream-reference.jsonl").read_text().splitlines(),
        )
    }
    cache = out / "base"

    held_to_fold = {}
    fit_sha = {}
    for index, fold in enumerate(split):
        for user in fold["held"]:
            if user in held_to_fold:
                raise ValueError(f"user {user}: held by more than one fold")
            held_to_fold[user] = index

        folder = out / f"fold-{index}"
        fit_path = folder / "fit.json"
        fit = json.loads(fit_path.read_text())
        if fit.get("split") != fold:
            raise ValueError(f"fold {index}: fitted split differs")

        expected_dependency = dependency_digest(cache, fold)
        if fit.get("base_dependency_sha256") != expected_dependency:
            raise ValueError(f"fold {index}: training/validation caches changed")

        model = folder / Path(fit["model"]).name
        if digest(model) != fit.get("model_sha256"):
            raise ValueError(f"fold {index}: saved booster changed")
        fit_sha[index] = digest(fit_path)

    if sorted(held_to_fold) != users:
        raise ValueError("stored held folds do not match evaluation users")

    total_reviews = 0
    for user in users:
        path = out / "scores" / f"{user}.json"
        if not path.is_file():
            raise ValueError(f"user {user}: missing score file")
        score = json.loads(path.read_text())

        sidecar = baseline.read_sidecar(cache, user)
        if sidecar.get("status") != "ok" or not sidecar.get("cache_sha256"):
            raise ValueError(f"user {user}: baseline provenance is not complete")

        fold = held_to_fold[user]
        expected_identity = {
            "user": user,
            "fold": fold,
            "base_sha256": sidecar["cache_sha256"],
            "fit_sha256": fit_sha[fold],
        }
        if score.get("identity") != expected_identity:
            raise ValueError(f"user {user}: stale score provenance")

        records = score.get("records")
        if not isinstance(records, dict) or set(records) != set(MODELS):
            raise ValueError(f"user {user}: score model set differs")

        if user not in reference:
            raise ValueError(f"user {user}: absent from upstream reference")
        expected_size = reference[user]["size"]
        for model in MODELS:
            record = records[model]
            if record.get("user") != user or record.get("size") != expected_size:
                raise ValueError(
                    f"user {user}: {model} coverage differs from upstream reference"
                )
        total_reviews += expected_size

    return total_reviews


def _parse_wrapper_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage",
        choices=["all", "baseline", "fit", "score", "report"],
        default="all",
    )
    parser.add_argument("--out", type=Path, default=HERE / "run")
    return parser.parse_known_args(argv)[0]


def _with_stage(argv: list[str], stage: str) -> list[str]:
    result = list(argv)
    if "--stage" in result:
        index = result.index("--stage")
        if index + 1 >= len(result):
            raise ValueError("--stage requires a value")
        result[index + 1] = stage
    else:
        result += ["--stage", stage]
    return result


def _run_frozen(argv: list[str], stage: str) -> None:
    subprocess.run(
        [sys.executable, str(FROZEN_RUNNER), *_with_stage(argv, stage)],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_wrapper_args(argv)

    if args.stage == "all":
        for stage in ("baseline", "fit", "score"):
            _run_frozen(argv, stage)
        total = validate_report_inputs(args.out)
        print(
            f"report inputs verified: "
            f"{len(json.loads((args.out / 'manifest.json').read_text())['users'])} "
            f"users / {total} rows"
        )
        _run_frozen(argv, "report")
        return 0

    if args.stage == "report":
        total = validate_report_inputs(args.out)
        print(
            f"report inputs verified: "
            f"{len(json.loads((args.out / 'manifest.json').read_text())['users'])} "
            f"users / {total} rows"
        )
        _run_frozen(argv, "report")
        return 0

    _run_frozen(argv, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
