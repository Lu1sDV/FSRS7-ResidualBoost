"""Acceptance: held-user outcomes cannot influence fitted shared stages."""

import json
import sys
import pickle
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import nested


def synthetic(user):
    t = np.arange(48)
    rating = ((t + user) % 4 + 1).astype(float)
    return {
        "y": (rating > 1).astype(float),
        "p_fsrs": 0.65 + ((t + user) % 7) * 0.04,
        "review_th": t + 1,
        "fold": t // 10,
        "i": t // 5 + 1,
        "elapsed_days": (t % 4 + 1).astype(float),
        "lapse": t // 13,
        "nth_today": t % 8,
        "day_offset": t // 8,
        "card_id": t % 5,
        "deck_id": t % 5 // 2,
        "note_id": t % 5,
        "rating": rating,
        "elapsed_seconds": (t % 4 + 1) * 86400.0,
        "duration": (t % 6 + 1) * 900.0,
    }


def test_outer_split_is_disjoint_complete_and_deterministic():
    users = list(range(1, 31))
    first = nested.make_splits(users, seed=20260919)
    assert first == nested.make_splits(list(reversed(users)), seed=20260919)
    held = []
    for fold in first:
        tr, va, ho = map(set, (fold["train"], fold["val"], fold["held"]))
        assert tr and va and ho
        assert not (tr & va or tr & ho or va & ho)
        assert tr | va | ho == set(users)
        held.extend(ho)
    assert sorted(held) == users


def test_current_and_future_answers_do_not_change_current_features():
    original = synthetic(1)
    changed = {k: v.copy() for k, v in original.items()}
    changed["y"][20:] = 1 - changed["y"][20:]
    changed["rating"][20:] = 1
    changed["duration"][20:] = 999999
    a, pa = nested.build_features(original, np.array([0.1, 3.0, 2.0]))
    b, pb = nested.build_features(changed, np.array([0.1, 3.0, 2.0]))
    np.testing.assert_array_equal(a[:21], b[:21])
    np.testing.assert_array_equal(pa[:21], pb[:21])


def test_calibration_weights_users_not_review_counts():
    short = {k: v[:1].copy() for k, v in synthetic(1).items()}
    long = {k: np.resize(v, 120) for k, v in synthetic(2).items()}
    short["y"][:] = short["p_fsrs"][:] = 1.0
    long["y"][:] = long["p_fsrs"][:] = 0.0
    # Residual windows are zero; equal-user offsets are symmetric about zero.
    # Pooled-review fitting instead shifts the intercept toward the 120 failures.
    beta = nested.fit_calibration([1, 2], lambda u: short if u == 1 else long)["beta"]
    np.testing.assert_allclose(beta, np.zeros(3), atol=1e-6)


def test_outer_fit_never_reads_held_labels_and_is_invariant(tmp_path):
    users = list(range(1, 17))
    split = nested.make_splits(users, seed=20260919)[0]
    allowed = set(split["train"]) | set(split["val"])
    seen = []

    def load(user):
        assert user in allowed, f"held-out user {user} read during fitting"
        seen.append(user)
        return synthetic(user)

    first = nested.fit_outer(
        split["train"], split["val"], load, tmp_path / "a", rounds=4, threads=1
    )
    assert set(seen) == allowed
    seen.clear()

    def perturbed(user):
        if user not in allowed:
            data = synthetic(user)
            data["y"] = 1 - data["y"]
            raise AssertionError("held-out data entered fitting")
        return load(user)

    second = nested.fit_outer(
        split["train"], split["val"], perturbed, tmp_path / "b", rounds=4, threads=1
    )
    np.testing.assert_array_equal(first["beta"], second["beta"])
    assert Path(first["model"]).read_bytes() == Path(second["model"]).read_bytes()
    # Each training user's calibration must exclude itself and all validation users.
    for fit in first["calibration_fits"]:
        assert set(fit["fit_users"]) <= set(split["train"])
        assert not set(fit["apply_users"]) & set(fit["fit_users"])


def test_interrupted_outer_fit_resumes_without_changing_model(tmp_path, monkeypatch):
    split = nested.make_splits(list(range(1, 17)), seed=20260919)[0]
    train, val = split["train"], split["val"]
    load = synthetic
    full = nested.fit_outer(
        train,
        val,
        load,
        tmp_path / "full",
        rounds=25,
        threads=1,
        checkpoint_key="same-inputs",
    )

    original_train = nested.xgb.train

    class Interrupt(nested.xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch == 13:
                raise RuntimeError("simulated crash")
            return False

    def interrupted(*args, **kwargs):
        kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), Interrupt()]
        return original_train(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(nested.xgb, "train", interrupted)
        with pytest.raises(RuntimeError, match="simulated crash"):
            nested.fit_outer(
                train,
                val,
                load,
                tmp_path / "resume",
                rounds=25,
                threads=1,
                checkpoint_key="same-inputs",
            )
    assert (tmp_path / "resume" / "boost-checkpoint.pkl").is_file()
    with pytest.raises(ValueError, match="checkpoint.*inputs"):
        nested.fit_outer(
            train,
            val,
            load,
            tmp_path / "resume",
            rounds=25,
            threads=1,
            checkpoint_key="different-inputs",
        )
    resumed = nested.fit_outer(
        train,
        val,
        load,
        tmp_path / "resume",
        rounds=25,
        threads=1,
        checkpoint_key="same-inputs",
    )
    assert resumed["best_iteration"] == full["best_iteration"]
    assert Path(resumed["model"]).read_bytes() == Path(full["model"]).read_bytes()


def test_checkpoint_resume_preserves_early_stopping(tmp_path):
    xgb = nested.xgb
    rng = np.random.default_rng(20260919)
    features = rng.normal(size=(256, 3))
    labels = (features[:, 0] > 0).astype(float)
    train = xgb.DMatrix(features, label=labels)
    val = xgb.DMatrix(features, label=1 - labels)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.2,
        "seed": 42,
        "nthread": 1,
    }

    def fit(directory, rounds, model=None, callbacks=()):
        return xgb.train(
            params,
            train,
            num_boost_round=rounds,
            evals=[(val, "val")],
            xgb_model=model,
            verbose_eval=False,
            callbacks=[nested._CheckpointEarlyStopping(directory, 15), *callbacks],
        )

    full = fit(tmp_path / "full", 30)
    original = xgb.train(
        params,
        train,
        num_boost_round=30,
        evals=[(val, "val")],
        early_stopping_rounds=15,
        verbose_eval=False,
    )
    assert full.best_iteration == original.best_iteration
    assert full.num_boosted_rounds() == original.num_boosted_rounds()
    np.testing.assert_array_equal(full.predict(val), original.predict(val))

    class Interrupt(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch == 12:
                raise RuntimeError("simulated crash")
            return False

    with pytest.raises(RuntimeError, match="simulated crash"):
        fit(tmp_path / "resume", 30, callbacks=[Interrupt()])
    checkpoint = tmp_path / "resume" / "boost-checkpoint.pkl"
    assert checkpoint.is_file()
    saved = pickle.loads(checkpoint.read_bytes())
    resumed = fit(tmp_path / "resume", 30 - saved.num_boosted_rounds(), model=saved)
    assert resumed.best_iteration == full.best_iteration
    assert resumed.num_boosted_rounds() == full.num_boosted_rounds()
    np.testing.assert_array_equal(resumed.predict(val), full.predict(val))


def test_calibration_does_not_diverge_on_confident_mistakes():
    from sklearn.metrics import log_loss

    data = synthetic(1)
    data["y"] = np.tile([0.0, 1.0], 24)
    data["p_fsrs"] = np.full(48, 0.999999)
    fitted = nested.fit_calibration([1, 2], lambda u: data)
    _, prediction = nested.build_features(data, fitted["beta"])
    assert log_loss(data["y"], prediction) <= log_loss(data["y"], data["p_fsrs"])


def test_streaming_matches_batch_at_saturated_calibration(tmp_path):
    from serve import StreamScorer

    # A real tree distinguishes zero residual from a wrongly clipped 1e-6 residual.
    X = np.zeros((2, len(nested.FEATURE_NAMES)), np.float32)
    X[1, nested.FEATURE_NAMES.index("sB_last")] = 1e-6
    booster = nested.xgb.train(
        {
            "objective": "binary:logistic",
            "max_depth": 1,
            "min_child_weight": 0,
            "eta": 8,
            "lambda": 0,
            "base_score": 0.5,
            "nthread": 1,
        },
        nested.xgb.DMatrix(X, label=[0, 1], feature_names=nested.FEATURE_NAMES),
        num_boost_round=1,
    )
    model = tmp_path / "saturated.json"
    booster.save_model(model)
    data = {key: value[:2].copy() for key, value in synthetic(1).items()}
    data["y"][:] = 1
    data["p_fsrs"][:] = 0.5
    beta = [50.0, 0.0, 0.0]
    offline = nested.predict_user(data, beta, booster, threads=1)
    stream = StreamScorer(model)
    stream.begin_user(beta)
    online = []
    for t in range(2):
        row = {
            key: data[key][t].item()
            for key in (
                "y",
                "p_fsrs",
                "i",
                "elapsed_days",
                "lapse",
                "nth_today",
                "card_id",
                "deck_id",
                "note_id",
                "rating",
                "elapsed_seconds",
                "duration",
            )
        }
        online.append(stream.observe(day=int(data["day_offset"][t]), **row))
    np.testing.assert_allclose(online, offline, atol=1e-6, rtol=0)


def test_release_pack_refuses_partial_evaluation_without_writing(tmp_path):
    import json
    import os
    import subprocess
    import sys

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "manifest.json").write_text(json.dumps({"users": [1, 2], "rounds": 4}))
    output = tmp_path / "release"
    script = Path(__file__).resolve().parents[2] / "scripts/make_submission_pack.sh"
    result = subprocess.run(
        ["bash", str(script), str(output), "--results", str(partial)],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert not output.exists()
    assert not output.with_suffix(".zip").exists()


def test_release_pack_requires_explicit_opt_in_for_deferred_users(tmp_path):
    import json
    import os
    import subprocess
    import sys

    results = tmp_path / "deferred"
    results.mkdir()
    here = Path(__file__).resolve().parent
    frozen = json.loads((here / "users.json").read_text())
    users = sorted(
        int(u) for u in (frozen["user_ids"] if isinstance(frozen, dict) else frozen)
    )
    deferred = users[-4:]
    covered = users[:-4]
    (results / "manifest.json").write_text(
        json.dumps(
            {
                "users": covered,
                "frozen_users": len(users),
                "deferred_users": deferred,
                "defer_reason": "test",
                "rounds": 600,
            }
        )
    )
    (results / "results.json").write_text(
        json.dumps({"users": len(covered), "reviews": 0, "deferred_users": deferred})
    )
    output = tmp_path / "release"
    script = Path(__file__).resolve().parents[2] / "scripts/make_submission_pack.sh"
    result = subprocess.run(
        ["bash", str(script), str(output), "--results", str(results)],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert "--allow-deferred" in result.stderr
    assert not output.exists()


def test_fold_selection_filters_and_validates():
    from run import select_folds

    assert select_folds(3, None) == [0, 1, 2]
    assert select_folds(3, [2]) == [2]
    assert select_folds(3, [2, 0]) == [0, 2]
    with pytest.raises(ValueError, match="fold"):
        select_folds(2, [2])
    with pytest.raises(ValueError, match="fold"):
        select_folds(2, [0, 0])


def test_score_shards_cover_every_held_user_exactly_once():
    import argparse

    from run import parse_shard, select_shard

    held = list(range(101))
    parts = [select_shard(held, (index, 8)) for index in range(8)]
    assert sum(len(part) for part in parts) == len(held)
    assert sorted(user for part in parts for user in part) == held
    assert select_shard(held, None) == held
    assert parse_shard("3/8") == (3, 8)
    for bad in ("x/y", "9/8", "-1/8", "0/0"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_shard(bad)



def _published_contract():
    root = Path(__file__).resolve().parents[2]
    here = root / "evaluation/submission-20260919"
    manifest = json.loads((here / "run/manifest.json").read_text())
    report = json.loads((here / "run/results.json").read_text())
    return root, manifest, report


def test_verifier_frozen_contract_accepts_published_release():
    import verify_submission

    root, manifest, report = _published_contract()
    protocol, users, deferred = verify_submission.validate_frozen_contract(
        root,
        manifest,
        report,
        qa_only=False,
    )
    assert len(users) == 9999
    assert deferred == []
    assert report["reviews"] == protocol["expected_scored_reviews"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("wrong-user-set", "frozen population"),
        ("wrong-review-count", "expected_scored_reviews"),
        ("missing-model", "model set"),
        ("wrong-round-count", "run rounds"),
        ("wrong-reference-hash", "reference_sha256"),
    ],
)
def test_verifier_frozen_contract_rejects_same_size_but_wrong_claims(
    mutation,
    message,
):
    import verify_submission

    root, manifest, report = _published_contract()
    manifest = json.loads(json.dumps(manifest))
    report = json.loads(json.dumps(report))

    if mutation == "wrong-user-set":
        manifest["users"][-1] = max(manifest["users"]) + 1
    elif mutation == "wrong-review-count":
        report["reviews"] -= 1
    elif mutation == "missing-model":
        report["models"].pop("B")
    elif mutation == "wrong-round-count":
        manifest["rounds"] -= 1
    elif mutation == "wrong-reference-hash":
        manifest["reference_sha256"] = "0" * 64
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        verify_submission.validate_frozen_contract(
            root,
            manifest,
            report,
            qa_only=False,
        )
