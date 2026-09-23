# Bias, leakage and submission-standards audit

## Decision

The packaged result is suitable for maintainer review **only as a nested refit on a previously studied public benchmark**. It is not a prospective untouched-data result, a scheduler evaluation, or evidence that the algorithm is already integrated into `srs-benchmark`.

## Corrected evaluation boundary

- Every reported user is held by one of two outer user folds. That user's reviews are not read by calibration or booster fitting for its prediction model.
- Booster-training users receive inner-cross-fitted calibration predictions: each training user's three calibration coefficients are fitted without that user. Validation and outer-held users use coefficients fitted only on the outer booster-training users.
- Booster early stopping reads outer validation users, never outer-held users. Each user is outer-held exactly once.
- Equal-user calibration and booster objectives prevent long review streams from silently dominating fitting. Reported summaries are equal-user means.
- Predictions are made before updating residual, rating, deck, note, card and timing state with the current review. State carries across the five upstream chronological test folds because the deployed predictor would carry history across time.

These properties are checked by `test_submission.py`, recorded in each fold's calibration lineage, and rechecked by `verify_submission.py` without raw review data.

## Baseline provenance

The base is freshly generated from unmodified `script.process` at upstream commit `bd9110f791e5b37282c55a9aa8db35f68f0c4aa2` with the frozen flags in `protocol.json`. Only persistence is replaced to retain unrounded evaluated rows. An independent stock-CLI comparison on user 1 found exact equality for all 10,620 labels, probabilities, fold memberships and causal fields; all nine upstream metrics and fitted parameters also matched. Cache identities cover upstream source bytes, capture source, dataset partition hashes, runtime versions, flags and user ID. Corruption or identity drift is rejected.

## Remaining bias and claim limits

1. **Prior model-selection exposure.** This public 10k-user dataset informed earlier feature and model-family development. Outer refitting removes outcome leakage from the reported fit; it cannot make prior development exposure disappear. The result may therefore be optimistic relative to a new population.
2. **No prospective external cohort.** The two outer folds cover the same previously studied corpus. A genuinely prospective claim requires a separately sourced cohort whose outcomes were unavailable throughout development.
3. **Instantaneous prediction only.** LogLoss, RMSE(bins) and AUC evaluate immediate recall predictions. They do not establish long-horizon retention, workload or scheduling gains.
4. **User-level target population.** Equal-user means answer performance for an average eligible user, not an average review. Per-review weighting would answer a different question.
5. **Undefined AUC.** AUC is null for users with one observed class. Reports state the non-null user count; LogLoss and RMSE(bins) require complete user coverage.
6. **Two outer folds.** Every user is held once, but there are only two fitted outer models. The reported per-user uncertainty does not capture model-selection uncertainty or population shift.
7. **Thread count is not bit-fixed.** The frozen protocol fixes booster hyperparameters but not execution thread count. Each fitted fold records its exact `nthread` in `fit.json`. Multi-threaded histogram reduction can change model bytes across thread counts; the baseline streams, calibration fits and split lineage are unaffected.
8. **Metadata snapshot.** Deck/note/card features reflect the distributed cards snapshot. Missing card metadata receives a unique fallback group rather than being merged into one artificial group.

## Distribution and reproducibility standard

The release contains source, the exact VPS orchestration script (`scripts/fullrun-chain-vps.sh`), two fitted boosters, calibration/lineage records, aggregate per-user metrics and provenance. It excludes raw reviews, card metadata, per-review predictions and feature matrices under the dataset's no-public-redistribution notice. Upstream source is pinned and its recorded sparse-checkout exclusions are reproducible from the README, but it is obtained separately and is not relicensed. The archive has full-file SHA-256 coverage, portable artifact paths, no source-checkout path, a detached ZIP checksum, and a verifier executed both before compression and after fresh extraction.

## Historical gap resolved in the new run

The historical `run-final/` deferred user **6810** (1,939,325 scored reviews). The other three users above 700K rows were completed there: 5859 (710,160), 6701 (1,611,815) and 8902 (1,342,505). That 9,998-user fit cannot be extended by adding 6810 and rerunning `report`: `nested.make_splits()` reshuffles both outer folds when the population size changes.

An earlier solo attempt on 6810 exceeded the local workstation's 24 GiB cgroup cap. A separate full-population fit was killed by host-wide OOM on the 30 GiB workstation at 2026-09-23 10:29:50; the baseline caches survived. The completed `run-vps/` evaluation used all **9,999** matching caches (including 6810), refitted both outer folds from scratch, scored **349,923,850** reviews, and produced a full report with **no deferred users** on a private 62 GiB host. The resulting equal-user LogLoss is **0.298813** versus **0.336968** for FSRS-7 on the same held users.

The upstream benchmark's published **RWKV-Instant** score is **0.2773** on the no-same-day table; this submission does not claim to beat that model or to be integrated into the upstream algorithm registry. The public dataset informed prior feature/model selection, so nested fitting removes fitting leakage but does not supply a prospective unseen-population validation.
