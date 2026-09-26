# FSRS7-ResidualBoost

Nested evaluation of an instantaneous recall predictor built from:

1. current upstream FSRS-7,
2. a three-coefficient equal-user residual calibration,
3. a 42-feature XGBoost correction.

This run fixes the historical cross-stage calibration leakage and does not reuse historical fitted calibration coefficients or boosters.

## Result

Frozen evaluation: **9,999 users**, **349,923,850 scored reviews**, no deferrals.

- upstream: `bd9110f791e5b37282c55a9aa8db35f68f0c4aa2`
- dataset: `75299740cff05894ef42d7ad990666691efdd2da`
- official summaries: equal-user means

| Model | LogLoss (99% CI half-width) | RMSE(bins) | AUC |
|---|---:|---:|---:|
| FSRS-7 | 0.336968 (0.004202) | 0.059287 | 0.722016 |
| B | 0.311472 (0.003804) | 0.041366 | 0.768187 |
| FSRS7-ResidualBoost | **0.298813 (0.003724)** | **0.033131** | **0.794077** |

AUC is defined for 9,938 users. Full results and per-user aggregates are in `evaluation/submission-20260919/run/`.

## Protocol

`protocol.json` is the frozen pre-run specification.

- Fresh per-user FSRS-7 predictions come from upstream `script.process` with `--short --secs --recency --equalize_test_with_non_secs`.
- Two deterministic outer user folds hold every evaluated user exactly once.
- Booster-training rows use three-way inner-cross-fitted calibration.
- Validation and held users use calibration fitted only on booster-training users.
- XGBoost uses equal-user weighting and early stopping on validation users only.
- Features are causal with respect to review outcomes: prediction happens before the current review updates residual/rating/timing/group state.
- Scored-review state carries across upstream chronological test-fold boundaries.
- Metrics use pinned upstream evaluation code.

The completed fit used eight XGBoost threads per outer fold. Thread count is recorded in each `fit.json`; multi-threaded histogram training is not guaranteed bit-identical across thread counts.

## Scope

This is a **nested refit on a previously studied public benchmark**.

It is not:

- a prospective untouched-population result;
- a scheduler-efficiency or workload evaluation;
- an already integrated upstream algorithm.

Deck/note grouping uses the distributed cards metadata snapshot. Historical availability of those group assignments is not independently established.

See `AUDIT.md` for the leakage, bias, provenance and claim-boundary review.

## Reproduction

Obtain the gated dataset directly from:

`https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k`

Use revision `75299740cff05894ef42d7ad990666691efdd2da`. Do not redistribute the raw dataset.

Reconstruct the recorded sparse upstream checkout:

```sh
git clone --filter=blob:none --no-checkout \
  https://github.com/open-spaced-repetition/srs-benchmark \
  evaluation/submission-20260919/upstream

git -C evaluation/submission-20260919/upstream sparse-checkout set --no-cone \
  '/*' '!/plots/' '!/result-20k/' '!/notebook/' '!/weights/' \
  '!/pretrain/' '!/result/' '!/rwkv/' '!/reptile/'

git -C evaluation/submission-20260919/upstream checkout \
  bd9110f791e5b37282c55a9aa8db35f68f0c4aa2

cd evaluation/submission-20260919/upstream
uv sync --frozen --dev --no-install-project --python 3.14
uv pip install --python .venv/bin/python \
  xgboost==3.4.1 pyrefly-shape-extensions==1.2.0
cd ../../..
```

Then run into a **new output directory**:

```sh
evaluation/submission-20260919/upstream/.venv/bin/python \
  evaluation/submission-20260919/run_hardened.py \
  --upstream evaluation/submission-20260919/upstream \
  --data /path/to/anki-revlogs-10k \
  --out evaluation/submission-20260919/reproduction-run \
  --workers 1 --threads 4
```

`--stage baseline|fit|score|report` can be used for resumable runs.

Do not use the bundled `run/` directory as `--out`; it contains the published, path-redacted artifacts.

For the completed full-run orchestration, see `scripts/fullrun-chain-vps.sh`.

`run.py` and `serve.py` are retained unchanged because they are part of the published run's provenance. For new report generation use `run_hardened.py`; for streaming use `serve_hardened.py`.

## Verification

For a normal Git checkout:

```sh
python scripts/verify_submission_checkout.py
```

To also verify the separately cloned upstream source identity:

```sh
python scripts/verify_submission_checkout.py \
  --upstream evaluation/submission-20260919/upstream
```

For a freshly extracted release archive, create the verification environment outside the archive:

```sh
uv venv ../audit-venv --python 3.14
uv pip install --python ../audit-venv/bin/python -r requirements.txt
../audit-venv/bin/python \
  evaluation/submission-20260919/verify_submission.py --root .
```

Behavioral tests:

```sh
evaluation/submission-20260919/upstream/.venv/bin/python -m pytest -q \
  evaluation/submission-20260919/test_submission.py \
  evaluation/submission-20260919/test_baseline.py \
  evaluation/submission-20260919/test_hardening.py
```

The verifier checks release checksums, frozen source/model identities, user coverage, split/calibration lineage, aggregate provenance and batch/streaming inference parity.

## Packaging

The distributable archive excludes upstream source, raw reviews, card metadata, per-review prediction caches and feature matrices.

The packager needs the private per-user `base/*.json` and `scores/*.json` provenance from the completed run. The large `base/*.npz` caches are not needed for packaging.

```sh
mkdir -p temp/run-vps-package-input
rsync -a --exclude='/base/*.npz' \
  evaluation/submission-20260919/run-vps/ \
  temp/run-vps-package-input/

PYTHON=evaluation/submission-20260919/upstream/.venv/bin/python \
  bash scripts/make_submission_pack.sh \
  "$HOME/fsrs7-residualboost-submission" \
  --results temp/run-vps-package-input
```

By default the packager refuses partial coverage or changed training sources, verifies staging and a fresh extraction, and writes a detached ZIP SHA-256.

`--allow-deferred` is only for visibly labeled partial archives. `--allow-smoke` is QA-only.

## Historical partial run

`run-final/` was a 9,998-user historical partial run that deferred user 6810. It is not the published full result and cannot be extended with a report-only rerun because changing the user population changes the outer splits.

The completed result refit both outer models from scratch over the full 9,999-user population.

## Licensing

Project contributions and trained artifacts are MIT-licensed to the extent rights are held. See `LICENSE` and `THIRD-PARTY-NOTICES.md`.

The dataset and upstream benchmark keep their own terms. Raw reviews and per-review caches must not be redistributed.
