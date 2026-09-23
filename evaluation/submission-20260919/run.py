#!/usr/bin/env python
"""Fresh current-upstream base plus two-fold, nested residual-boost evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_user_ids(path):
    """Frozen population as a plain sorted id list, accepting list or {"user_ids": ...}."""
    payload = json.loads(Path(path).read_text())
    ids = payload["user_ids"] if isinstance(payload, dict) else payload
    return sorted(int(u) for u in ids)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # pid-scoped temp: concurrent runner processes may save the same path (manifest,
    # splits, users) during startup; a shared fixed temp name could interleave writes.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_folds(count, requested):
    """Fold indices owned by this runner process; None means every fold."""
    if requested is None:
        return list(range(count))
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate fold selection: {requested}")
    invalid = sorted(set(requested) - set(range(count)))
    if invalid:
        raise ValueError(f"unknown fold {invalid} of {count} folds")
    return sorted(requested)


def select_shard(items, shard):
    """Slice of items owned by one score process; None means all of them."""
    if shard is None:
        return list(items)
    index, count = shard
    return [item for position, item in enumerate(items) if position % count == index]


def parse_shard(value):
    """Parse `I/N` (0-based) for one score process."""
    try:
        index_text, count_text = value.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shard must be I/N with integers") from error
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"shard must satisfy 0 <= I < N, got {value}")
    return index, count


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def upstream_metrics(upstream, data):
    sys.path.insert(0, str(upstream))
    upstream_config = importlib.import_module("config")
    evaluate = importlib.import_module("utils").evaluate
    flags = json.loads((HERE / "protocol.json").read_text())["base_flags"]
    config = upstream_config.Config(
        upstream_config.create_parser().parse_args([*flags, "--data", str(data)])
    )
    return evaluate, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "baseline", "fit", "score", "report"], default="all"
    )
    parser.add_argument("--upstream", type=Path, default=HERE / "upstream")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--users", type=Path, default=HERE / "users.json")
    parser.add_argument("--out", type=Path, default=HERE / "run")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="fit stage: fold indices this process owns (folds may run in parallel)",
    )
    parser.add_argument(
        "--shard",
        type=parse_shard,
        default=None,
        help="score stage: I/N slice of held users this process owns",
    )
    parser.add_argument(
        "--defer-users",
        type=Path,
        help="JSON user-id list excluded from the frozen population and reported as an explicit coverage gap",
    )
    parser.add_argument(
        "--defer-reason",
        default="excluded from this run; requires more memory than the executing host provides",
    )
    args = parser.parse_args()
    if min(args.workers, args.threads, args.rounds) < 1:
        parser.error("workers, threads and rounds must be positive")
    if args.folds is not None and args.stage != "fit":
        parser.error("--folds applies only to --stage fit")
    if args.shard is not None and args.stage != "score":
        parser.error("--shard applies only to --stage score")
    import baseline
    import nested

    users = json.loads(args.users.read_text())
    users = users["user_ids"] if isinstance(users, dict) else users
    users = sorted(int(u) for u in users)
    if len(users) != len(set(users)) or len(users) < 12:
        parser.error(
            "need at least 12 unique users for two outer folds and nested calibration"
        )
    deferred = []
    if args.defer_users is not None:
        declared = json.loads(args.defer_users.read_text())
        deferred = sorted(
            int(u)
            for u in (declared["user_ids"] if isinstance(declared, dict) else declared)
        )
        unknown = set(deferred) - set(users)
        if unknown:
            parser.error(
                f"deferred users are not in the frozen population: {sorted(unknown)}"
            )
        users = [u for u in users if u not in set(deferred)]
        if len(users) < 12:
            parser.error("too few users remain after deferrals")
    args.out = args.out.resolve()
    args.upstream = args.upstream.resolve()
    args.data = args.data.resolve()
    protocol = json.loads((HERE / "protocol.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != protocol["upstream_revision"]:
        raise ValueError(
            f"upstream revision {revision} != frozen {protocol['upstream_revision']}"
        )
    sources = [
        HERE / f for f in ("run.py", "baseline.py", "nested.py", "protocol.json")
    ]
    sources += [
        HERE.parent / "tabular-probe-20260917" / f for f in ("probe.py", "serve.py")
    ]
    sources += [HERE.parent / "probe-stage2-20260918/stage2.py"]
    base_runtime = baseline.env_versions()
    manifest = {
        "upstream_revision": revision,
        "upstream_source": baseline.source_identity(args.upstream),
        "runtime_versions": {**base_runtime, "xgboost": xgb.__version__},
        "reference_sha256": digest(HERE / "upstream-reference.jsonl"),
        "dataset_revision": protocol["dataset_revision"],
        "data": str(args.data),
        "users": users,
        "frozen_users": len(read_user_ids(HERE / "users.json")),
        "deferred_users": deferred,
        "defer_reason": args.defer_reason if deferred else None,
        "rounds": args.rounds,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "protocol_sha256": digest(HERE / "protocol.json"),
        "evaluation_kind": "nested refit on previously studied public benchmark",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "run manifest differs: use a new output directory, never mix runs"
        )
    save(manifest_path, manifest)
    split = nested.make_splits(users, seed=protocol["split_seed"])
    owned_folds = select_folds(len(split), args.folds)
    split_path = args.out / "splits.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError("stored splits differ from frozen split rule")
    save(split_path, split)
    effective_users_path = args.out / "users.json"
    save(effective_users_path, users)
    stages = (
        ["baseline", "fit", "score", "report"] if args.stage == "all" else [args.stage]
    )
    cache = args.out / "base"
    reference = {
        r["user"]: r
        for r in map(
            json.loads, (HERE / "upstream-reference.jsonl").read_text().splitlines()
        )
    }

    def load(user):
        sidecar = baseline.read_sidecar(cache, user)
        expected = baseline.identity_fields(
            args.upstream,
            baseline.OFFICIAL_FLAGS,
            protocol["dataset_revision"],
            user,
            baseline.data_files(args.data, user),
            runtime=base_runtime,
        )
        if sidecar["identity"] != baseline.identity_digest(expected):
            raise ValueError(
                f"user {user}: base cache is not from this source/data/runtime"
            )
        result = baseline.load_user(cache, user)
        if user not in reference or len(result["y"]) != reference[user]["size"]:
            raise ValueError(
                f"user {user}: scored row count differs from current upstream reference"
            )
        return result

    def dependency_digest(fold):
        dependencies = {}
        for user in sorted(fold["train"] + fold["val"]):
            sidecar = baseline.read_sidecar(cache, user)
            dependencies[str(user)] = [sidecar["identity"], sidecar["cache_sha256"]]
        return hashlib.sha256(
            json.dumps(dependencies, sort_keys=True).encode()
        ).hexdigest()

    for stage in stages:
        print(f"phase={stage} users={len(users)}", flush=True)
        if stage == "baseline":
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "baseline.py"),
                    "--upstream",
                    str(args.upstream),
                    "--data",
                    str(args.data),
                    "--out",
                    str(cache),
                    "--users",
                    str(effective_users_path),
                    "--workers",
                    str(args.workers),
                    "--dataset-revision",
                    protocol["dataset_revision"],
                    "--python",
                    sys.executable,
                ],
                check=True,
            )
            for user in users:
                load(user)
        elif stage == "fit":
            for index in owned_folds:
                fold = split[index]
                folder = args.out / f"fold-{index}"
                expected_dependency = dependency_digest(fold)
                fit_path = folder / "fit.json"
                if fit_path.exists():
                    fit = json.loads(fit_path.read_text())
                    model = folder / Path(fit["model"]).name
                    if fit["base_dependency_sha256"] != expected_dependency:
                        raise ValueError(
                            f"fold {index}: training/validation caches changed"
                        )
                    if digest(model) != fit["model_sha256"]:
                        raise ValueError(f"fold {index}: saved booster changed")
                    continue
                print(
                    f"fit fold={index} train={len(fold['train'])} val={len(fold['val'])} held={len(fold['held'])}",
                    flush=True,
                )
                fit = nested.fit_outer(
                    fold["train"],
                    fold["val"],
                    load,
                    folder,
                    rounds=args.rounds,
                    threads=args.threads,
                    checkpoint_key=expected_dependency,
                )
                fit["model_sha256"] = digest(fit["model"])
                fit["split"] = fold
                fit["base_dependency_sha256"] = expected_dependency
                save(fit_path, json_safe(fit))
        elif stage == "score":
            evaluate, config = upstream_metrics(args.upstream, args.data)
            scores = args.out / "scores"
            scores.mkdir(exist_ok=True)
            for index, fold in enumerate(split):
                folder = args.out / f"fold-{index}"
                fit = json.loads((folder / "fit.json").read_text())
                if fit["base_dependency_sha256"] != dependency_digest(fold):
                    raise ValueError(
                        f"fold {index}: training/validation caches changed"
                    )
                model = folder / Path(fit["model"]).name
                if digest(model) != fit["model_sha256"]:
                    raise ValueError(f"fold {index}: saved booster changed")
                booster = xgb.Booster(model_file=str(model))
                booster.set_param({"nthread": args.threads})
                owned = select_shard(fold["held"], args.shard)
                for n, user in enumerate(owned, 1):
                    target = scores / f"{user}.json"
                    data = load(user)
                    identity = {
                        "user": user,
                        "fold": index,
                        "base_sha256": digest(cache / f"{user}.npz"),
                        "fit_sha256": digest(folder / "fit.json"),
                    }
                    if target.exists():
                        old = json.loads(target.read_text())
                        if old["identity"] != identity:
                            raise ValueError(f"user {user}: result provenance mismatch")
                        continue
                    X, pb = nested.build_features(data, np.asarray(fit["beta"]))
                    margin = np.log(
                        np.clip(pb, 1e-6, 1 - 1e-6) / (1 - np.clip(pb, 1e-6, 1 - 1e-6))
                    )
                    prediction = booster.predict(
                        xgb.DMatrix(
                            X,
                            base_margin=margin.astype(np.float32),
                            feature_names=booster.feature_names,
                        ),
                        iteration_range=(0, booster.best_iteration + 1),
                    )
                    records = {}
                    for name, probability in [
                        ("FSRS-7", data["p_fsrs"]),
                        ("B", pb),
                        ("FSRS7-ResidualBoost", prediction),
                    ]:
                        if not np.isfinite(probability).all() or np.any(
                            (probability < 0) | (probability > 1)
                        ):
                            raise ValueError(f"user {user}: invalid {name} probability")
                        frame = pd.DataFrame(
                            {
                                "y": data["y"],
                                "p": probability,
                                "elapsed_days": data["elapsed_days"],
                                "i": data["i"],
                                "rmse_bins_lapse": data["lapse"],
                            }
                        )
                        record, _ = evaluate(
                            data["y"], probability, frame, name, user, config
                        )
                        records[name] = json_safe(record)
                    save(target, {"identity": identity, "records": records})
                    if n % 25 == 0 or n == len(owned):
                        print(
                            f"score fold={index} complete={n}/{len(owned)} user={user}",
                            flush=True,
                        )
        else:
            result_paths = [args.out / "scores" / f"{u}.json" for u in users]
            missing = [u for u, p in zip(users, result_paths) if not p.exists()]
            if missing:
                raise ValueError(f"incomplete evaluation: {len(missing)} users missing")
            rows = [json.loads(p.read_text()) for p in result_paths]
            metric_file = args.upstream / "evaluate.py"
            spec = importlib.util.spec_from_file_location(
                "upstream_aggregation", metric_file
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot load pinned upstream aggregation")
            aggregation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(aggregation)
            report = {
                "users": len(users),
                "reviews": sum(reference[u]["size"] for u in users),
                "source": "new nested fit; historical results not reused",
                "deferred_users": deferred,
                "defer_reason": args.defer_reason if deferred else None,
                "models": {},
            }
            for model in ("FSRS-7", "B", "FSRS7-ResidualBoost"):
                records = [row["records"][model] for row in rows]
                if [r["user"] for r in records] != users:
                    raise ValueError("record identity differs from frozen users")
                (args.out / f"result-{model}.jsonl").write_text(
                    "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
                )
                metrics = {}
                for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                    values = np.array(
                        [
                            r["metrics"][metric]
                            for r in records
                            if r["metrics"][metric] is not None
                        ],
                        float,
                    )
                    if metric != "AUC" and len(values) != len(users):
                        raise ValueError(f"{model} {metric}: missing values")
                    if not np.isfinite(values).all():
                        raise ValueError(f"{model} {metric}: nonfinite values")
                    metrics[metric] = {
                        "mean": float(values.mean()),
                        "n_users": len(values),
                        "ci99_half_width": float(
                            aggregation.confidence_interval(
                                values, np.ones(len(values))
                            )
                        ),
                    }
                report["models"][model] = metrics
            save(args.out / "results.json", report)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
