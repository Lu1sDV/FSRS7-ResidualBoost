# FSRS7-ResidualBoost: nested current-upstream evaluation

A composite instantaneous recall predictor: current upstream FSRS-7, a three-coefficient equal-user residual calibration, then a 42-feature XGBoost correction. This implementation corrects the historical evaluation's cross-stage calibration leakage. It does not reuse historical fitted calibration coefficients, boosters or base prediction caches.

## Frozen protocol

`protocol.json` is the pre-run specification. `users.json` identifies the 9,999 eligible users from current upstream's non-same-day benchmark (349,923,850 scored reviews). Upstream revision: `bd9110f791e5b37282c55a9aa8db35f68f0c4aa2`. Dataset revision: `75299740cff05894ef42d7ad990666691efdd2da`.

- Fresh per-user chronological FSRS-7 training uses upstream `script.process` with `--short --secs --recency --equalize_test_with_non_secs`. Only its persistence callback is replaced, preserving unrounded probabilities, scored review identities and official metrics.
- Two deterministic outer user folds cover each user once. Each fold divides the remaining users into booster training and validation without consulting outcomes.
- Training-row calibration is inner-cross-fitted over three training-user groups. Validation and held-user predictions use calibration fitted only on booster training users. Validation/held outcomes never enter these calibration fits.
- XGBoost trains on equal-user weights scaled to mean one over the whole training partition; validation uses independently normalized equal-user weights. Feature definitions/hyperparameters are frozen, with early stopping on validation users only. The completed VPS fit used `--threads 8` per outer fold; thread count is a runner parameter, not a frozen hyperparameter, and each fold's `fit.json` records the exact `nthread` used. As with any multi-threaded histogram build, the fitted model is reproducible in distribution but not guaranteed bit-identical across thread counts.
- Scored reviews update state after prediction, carrying history across chronological base-fold boundaries. Unscored reviews do not update the correction's history. Current upstream supplies the RMSE-bin lapse definition.
- Current upstream `utils.evaluate` computes the result records, and upstream `evaluate.confidence_interval` provides equal-user 99% BCa intervals using 9,999 bootstrap resamples with `random_state=42`. No partial coverage is reported as a full benchmark.

The artifact is an **instantaneous predictor**, not a validated interval scheduler. This is a nested refit on a public dataset used during prior development, not a claim of prospective performance on never-examined users. Earlier exposure, metadata snapshot limitations and model-selection uncertainty remain disclosed scientific limitations rather than hidden exclusions. The old scores do not transfer to this protocol.

See `AUDIT.md` for the independent leakage, bias, provenance and claim-boundary assessment.

## Completed full-population result (2026-09-23)

The full run completed with **9,999/9,999 users**, **349,923,850 scored reviews**, and no deferrals, using upstream `bd9110f791e5b37282c55a9aa8db35f68f0c4aa2`. Equal-user means are the official benchmark metric; review-weighted means are included as supplementary context. The official 99% BCa confidence intervals use 9,999 bootstrap resamples with `random_state=42`.

| Model | Equal-user LogLoss (99% CI half-width) | Review-weighted LogLoss | Equal-user RMSE(bins) | Review-weighted RMSE(bins) | Equal-user AUC | Review-weighted AUC |
|---|---:|---:|---:|---:|---:|---:|
| FSRS-7 | 0.336968 (0.004202) | 0.315055 | 0.059287 | 0.040771 | 0.722016 | 0.721673 |
| B | 0.311472 (0.003804) | 0.298430 | 0.041366 | 0.028686 | 0.768187 | 0.758742 |
| FSRS7-ResidualBoost | **0.298813 (0.003724)** | **0.289265** | **0.033131** | **0.022837** | **0.794077** | **0.784481** |

Per-user LogLoss deltas against FSRS-7 (variant minus reference; negative is better), paired percentile bootstrap, 20,000 resamples with seed 0:

| Variant | Mean Δ LogLoss | Paired 95% bootstrap CI | Users improved / worsened / tied |
|---|---:|---:|---:|
| B | -0.025496 | [-0.026440, -0.024583] | 9,269 / 730 / 0 |
| FSRS7-ResidualBoost | -0.038155 | [-0.039346, -0.037010] | 9,919 / 80 / 0 |

AUC is undefined for 61 users; the review-weighted AUC denominator is 349,421,805 reviews over the remaining 9,938 users. Review-weighted values do not replace the equal-user official metric.

Each outer-fold model has three calibration coefficients and 600 XGBoost trees. Counting all split and leaf nodes in the serialized boosters: fold 0 has 72,458 nodes (72,461 calibration-coefficient-plus-node count); fold 1 has 73,326 nodes (73,329 combined count). Across both stored fold models: six calibration coefficients and 145,784 XGBoost nodes. This is a structural node count, not a leaf-weight-only parameter count.

**Execution record.** `/root/srs-autoresearch/temp/fullrun-chain-vps.sh` ran on a private 16-vCPU, 62-GiB VPS, with `CPUQuota=16`, `MemoryHigh=56 GiB`, `MemoryMax=58 GiB`, and swap disabled. It fit folds 0 and 1 concurrently at eight threads each, then scored eight shards at two threads each, then generated the report. Journal timestamps: 2026-09-23 15:28:44 to 21:34:09 CEST (**6 h 05 min 25 s wall time**, including the missing-dependency score retries). Systemd reported approximately **73 h 03 min of cumulative CPU time** for the main fit, the successful scoring/report service invocation, and short retries. The precomputed full-population base caches were reused and validated; these times do not cover baseline-cache generation.

This is a nested refit on a public dataset used during prior development, not a prospective unseen-dataset result. It is an instantaneous predictor, not evidence of improved scheduling. The upstream no-same-day benchmark currently reports **RWKV-Instant at 0.2773 LogLoss**; our 0.298813 is **not** a claim to outperform it. Our paired comparison is only against FSRS-7 on the same held users. Our cross-user training protocol is not yet integrated into the upstream algorithm registry.

## Reproduction

Obtain the dataset directly from https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k, complying with its access/no-redistribution terms. The runner expects `revlogs/user_id=<id>/data.parquet` and the cards metadata partitions under `--data`.

Obtain upstream separately; its source is not relicensed by this submission. The exercised setup was:

```sh
git clone --filter=blob:none https://github.com/open-spaced-repetition/srs-benchmark \
  evaluation/submission-20260919/upstream
git -C evaluation/submission-20260919/upstream checkout \
  bd9110f791e5b37282c55a9aa8db35f68f0c4aa2
cd evaluation/submission-20260919/upstream
uv sync --frozen --dev --no-install-project --python 3.14
uv pip install --python .venv/bin/python xgboost==3.4.1 pyrefly-shape-extensions==1.2.0
cd ../../..
evaluation/submission-20260919/upstream/.venv/bin/python \
  evaluation/submission-20260919/run.py \
  --upstream evaluation/submission-20260919/upstream \
  --data /path/to/anki-revlogs-10k \
  --out evaluation/submission-20260919/run --workers 1 --threads 4
```

`--stage baseline|fit|score|report` selects a resumable stage. The run manifest rejects source/protocol/user-set changes. Base caches require matching source/data provenance, hashes and chronological row identities. Training records the calibration dependency users; final predictions must use their designated outer fold. Report generation requires all frozen users. Use a new output directory for a changed experiment.

The full-corpus feature matrix is not materialized in RAM: per-user feature batches and XGBoost external memory keep the training path bounded. Raw/per-review caches are local reconstruction inputs and are excluded from distribution.

## Verify and package

The release archive omits upstream source, raw reviews and per-review caches. From an extracted archive, install `requirements.txt` and run:

```sh
python evaluation/submission-20260919/verify_submission.py --root .
```

The verifier checks every distributed file against `SHA256SUMS`, training-source hashes, model hashes, exact split/calibration lineage, per-user aggregate coverage and provenance, reported metric means, and batch/streaming inference parity. To assemble the final archive after a complete frozen run, transfer `run-vps/` into this checkout first. The packager reads each user's `base/*.json` provenance and `scores/*.json`, plus both fold models and report files; the large `base/*.npz` prediction caches are **not** needed for packaging and must not be published:

```sh
mkdir -p temp/run-vps-package-input
rsync -a --exclude='/base/*.npz' \
  evaluation/submission-20260919/run-vps/ temp/run-vps-package-input/
PYTHON=evaluation/submission-20260919/upstream/.venv/bin/python \
  bash scripts/make_submission_pack.sh "$HOME/fsrs7-residualboost-submission" \
    --results temp/run-vps-package-input
```

By default the packager refuses partial coverage or changed training sources, verifies staging and a fresh extraction, and writes a detached ZIP SHA-256. `--allow-deferred` explicitly permits a visibly labeled partial archive, **not** a full-table submission. `--allow-smoke` is QA-only and requires an output name ending in `-smoke`.

## Historical coverage gap

`run-final/` is a historical **partial** result, not the complete benchmark. One frozen user was deferred from that run:

| User | Scored reviews | Why |
|---|---:|---|
| 6810 | 1,939,325 | needs more than the 16 GiB batch-slice budget |

The other three users above 700K rows are covered: 5859 (710,160), 6701 (1,611,815) and 8902 (1,342,505). A solo attempt on 6810, with every other job stopped and a 24 GiB cgroup cap, reached 22.07 GiB before the host killed it at 28 GiB used / 291 MiB free on a 30 GiB machine.

The historical local batch slice was capped at 14 GiB with swap disabled. That did **not** prevent a host-wide OOM during the local full-population booster fit: at 2026-09-23 10:29:50 the kernel killed the ~11 GiB fit while desktop memory and swap were exhausted. The local service was stopped and disabled. The historical gap is now closed: all 9,999 verified base caches, including user 6810, were reused for the complete fit/score/report described above on the private VPS.

To reproduce the completed full-coverage refit, put **all 9,999 users' matching `.json` sidecars and `.npz` caches** into a new output directory's `base/` (including user 6810). Alternatively, the `baseline` stage generates missing caches and validates existing ones. Then refit both outer folds from scratch:

```sh
RUN=evaluation/submission-20260919/run-full
DATA=/path/to/anki-revlogs-10k
python evaluation/submission-20260919/run.py --out "$RUN" --data "$DATA" --stage baseline --workers 1
python evaluation/submission-20260919/run.py --out "$RUN" --data "$DATA" --stage fit --threads 8
python evaluation/submission-20260919/run.py --out "$RUN" --data "$DATA" --stage score --threads 2
python evaluation/submission-20260919/run.py --out "$RUN" --data "$DATA" --stage report
```

`--stage report` alone is **not** sufficient, and reusing `run-final/` is rejected (`run manifest differs`). `nested.make_splits()` shuffles with `rng.permutation(len(users))`, so adding 6810 moves **1,337 of the other 9,998 users** into a different outer fold and changes fold-0's training set by **2,123 users**. Every old fitted correction and booster is invalid for the full population.

## Behavioral acceptance

| Action | Observable outcome | Check |
|---|---|---|
| Fit an outer model with access restricted to training/validation users | Held-user outcomes are never read; perturbing their contents cannot change fitted calibration/model bytes | `test_submission.py` |
| Build features after changing current/future outcomes, grades or response times | Current and earlier features/predictions stay unchanged | `test_submission.py` |
| Predict with calibration saturated at zero or one | Streaming and batch predictions agree; logit clipping does not alter residual history | `test_submission.py` |
| Archive a run that defers frozen users | Refused unless `--allow-deferred` is passed; the archive then ships `COVERAGE-GAP.txt` naming every deferred user | `test_submission.py` |
| Split shuffled input user IDs | Deterministic disjoint train/validation/held partitions; every user held exactly once | `test_submission.py` |
| Resume a baseline cache with changed/corrupt inputs | Refuse the cache instead of mixing predictions | `test_baseline.py` |
| Interrupt an outer booster fit and restart with the same input identity | Resume from the ten-round snapshot with the same early-stopping outcome as an uninterrupted fit; changed identity is refused | `test_submission.py` |

## Licensing

The user authorized MIT for their own contributions and trained artifacts. See `LICENSE` and `THIRD-PARTY-NOTICES.md`. Raw reviews and per-review caches must not be included in a public submission. Upstream and dependencies keep their own applicable terms.
