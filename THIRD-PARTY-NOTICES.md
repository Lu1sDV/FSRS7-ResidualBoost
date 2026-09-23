# License scope and third-party notices

The user explicitly authorized MIT licensing on 2026-09-19 for their own contributions and trained artifacts, to the extent they hold the necessary rights. `LICENSE` applies to those contributions. It does not purport to relicense upstream benchmark code or the dataset.

## Upstream benchmark

https://github.com/open-spaced-repetition/srs-benchmark

Pinned revision: `bd9110f791e5b37282c55a9aa8db35f68f0c4aa2`.

Upstream code remains upstream's work under its applicable terms. The submission archive does not redistribute a full upstream source snapshot under our MIT grant: reproduction instructions obtain the upstream checkout separately. Source hashes and the pin identify what was run. Python/XGBoost/PyTorch and other dependencies retain their respective licenses.

## Dataset

https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k

Recorded revision: `75299740cff05894ef42d7ad990666691efdd2da`.

The downloaded dataset notice permits memory research by individuals and university students and prohibits public redistribution of the data. Obtain it directly from its source and obey its access terms. No raw reviews, card/deck snapshots, per-review prediction caches or feature matrices are included in the distributable archive. Trained model artifacts and aggregate benchmark results are research outputs, not a grant to redistribute underlying review data.

## Feature implementation

The 42-feature residual/rating/interval/deck/note recipe comes from this project's `tabular-probe-20260917` and `probe-stage2-20260918` research code. The corrected runner reuses that implementation rather than redefining the feature semantics. Those project contributions are covered by the user's MIT authorization; any retained third-party notices continue to apply.
