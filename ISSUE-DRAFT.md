# Proposed FSRS7-ResidualBoost benchmark evaluation

Coverage: 9,999 users / 349,923,850 scored reviews. Current upstream pin: `bd9110f791e5b37282c55a9aa8db35f68f0c4aa2`.

| Model | LogLoss | RMSE(bins) | AUC |
|---|---:|---:|---:|
| FSRS-7 | 0.336968 | 0.059287 | 0.722016 |
| B | 0.311472 | 0.041366 | 0.768187 |
| FSRS7-ResidualBoost | 0.298813 | 0.033131 | 0.794077 |

Two outer user folds; inner-cross-fitted three-coefficient calibration; equal-user boosted correction. Validation and held-user labels never fit the calibration used for those users. Fresh chronological upstream FSRS-7 predictions; no historical fitted corrections or boosters reused.

This fixes cross-stage fitting leakage, not prior model-selection exposure: the public dataset was studied previously. These are instantaneous recall predictions, not evidence of improved scheduling. This cross-user training protocol is supplied as a standalone runner, not represented as an already-merged upstream algorithm registration.

The archive contains MIT-authorized project source and trained artifacts, per-user aggregate results and provenance, SHA256 checksums, a data-free verifier, and complete reproduction commands. Dataset and upstream rights remain separate. Raw reviews and per-review caches are deliberately absent.
