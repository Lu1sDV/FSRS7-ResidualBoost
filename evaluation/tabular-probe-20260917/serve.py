#!/usr/bin/env python
"""Streaming inference for the tabular probe: one review in, `p_hat` out.

`probe.py` computes the 32 features vectorised over a user's whole history, which is fine
for research and useless for serving.  This module is the same feature set as an online
scorer: `begin_user()` per user, then `observe(...)` per review, which returns the
prediction for that review and afterwards advances the state with the review's own
residual.  It mirrors `probe.compute_features` exactly -- same windows, same shrinkage,
same causal deck/note state machine, same `rB`-based aggregates.

    from serve import StreamScorer, load_betas
    scorer, betas = StreamScorer("model-all.json"), load_betas()
    for user, reviews in stream:
        scorer.begin_user(beta=betas.get(user))
        p_hat = [scorer.observe(**r) for r in reviews]

Per review: `y` (this review's outcome, used only to update state after the fact),
`p_fsrs` (base-model probability), `i`, `elapsed_days`, `lapse`, `nth_today`, `day`,
`card_id`, `deck_id`, `note_id`.  `p_b` is optional: when omitted it is derived as in
`probe.derive_p_b` from `p_fsrs` plus the per-user B coefficients (`crossfit.json` group
fits), which is the transportable path the external panel used.

Models that carry the `stream` family (`model-foldin-all32_stream.json`, 42 features) also
need `rating`, `elapsed_seconds` and `duration` per review — the user's chosen button, the
card's Δt in seconds, and the response time in ms.  They are rejected with `None`
(default) rather than silently filled: a serving path that cannot supply them must not
pretend to.

Self-check (implementation parity against the persisted artifacts):

    python serve.py verify --users 16 21 32
    python serve.py verify --panel fresh --users 5001 5002 \
        --model ../probe-stage2-20260918/model-foldin-all32_stream.json \
        --stream-table ../probe-stage2-20260918/stage2-fresh.parquet \
        --pred-file ../stack-fresh-20260918/stream-preds-fresh.parquet --pred-col p_stream
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

HERE = Path(__file__).resolve().parent
CLIP = 1e-6
SHRINK = 64                      # n/(n+SHRINK), frozen by residual-decomposition-20260909
WINDOWS = (16, 256)              # spine windows
SURPRISE_WINDOWS = (16, 64)
ALL_WINDOWS = tuple(sorted(set(WINDOWS) | set(SURPRISE_WINDOWS)))  # 16, 64, 256
CROSSFIT = HERE.parent / "mixture-ab-20260916/sources/crossfit-eq/crossfit.json"
INPUT_COLUMNS = ["user", "row_index", "y", "p_fsrs", "p_b", "i_raw", "elapsed_days",
                 "lapse", "day", "card_id", "deck_id", "note_id", "st_nth_today"]
# The `stream` family (stage 2): user-level rating history + the card's sub-day timing.
# Definitions mirror `probe-stage2-20260918/stage2.py` exactly -- windows over *previous*
# reviews, 0.0 for the first row, the same 600 s same-session threshold.
STREAM_FEATS = ["rt_mean8", "rt_mean32", "rt_sd32", "rt_last", "rt_fail32",
                "rt_good_streak", "rt_lapse_streak", "sg_log_secs", "sg_same_session",
                "sg_prev_dur_log"]
RATING_WINDOW = 32
SAME_SESSION_SECONDS = 600.0


def feature_names(panel: str = "lockbox") -> list[str]:
    """Authoritative column order from the spec written by `probe.py spec`."""
    path = HERE / ("feature-spec.json" if panel == "lockbox" else "feature-spec-fresh.json")
    if not path.exists():
        path = HERE / "feature-spec.json"
    return list(json.loads(path.read_text())["features_flat"])


def load_betas(path: Path = CROSSFIT) -> dict[int, list[float]]:
    """Per-user B coefficients: `fits[g].coefficients` applies to `groups[g]`."""
    cf = json.loads(path.read_text())
    return {int(u): cf["fits"][g]["coefficients"]
            for g, grp in enumerate(cf["groups"]) for u in grp}


def _logit(p: float) -> float:
    p = min(max(float(p), CLIP), 1 - CLIP)
    return float(np.log(p / (1 - p)))


class StreamScorer:
    """Online feature build + booster for the probe's `all` variant (or any saved booster)."""

    def __init__(self, model_path: str | Path = HERE / "model-all.json",
                 panel: str = "lockbox"):
        self.bst = xgb.Booster()
        self.bst.load_model(str(model_path))
        self.names = list(self.bst.feature_names or feature_names(panel))
        self.stream = bool(set(self.names) & set(STREAM_FEATS))
        unknown = set(self.names) - set(feature_names(panel)) - set(STREAM_FEATS)
        if unknown:
            raise ValueError(f"model expects unknown features: {sorted(unknown)}")
        best = getattr(self.bst, "best_iteration", None)
        self.iteration_range = (0, best + 1) if best is not None else (0, 0)
        self.begin_user()

    # -- per-user state ------------------------------------------------------
    def begin_user(self, beta: list[float] | None = None) -> None:
        self.beta = beta
        self.t = 0
        self.win_F = {w: deque(maxlen=w) for w in ALL_WINDOWS}
        self.win_B = {w: deque(maxlen=w) for w in ALL_WINDOWS}
        self.win_F_max = {w: deque(maxlen=w) for w in SURPRISE_WINDOWS}
        self.win_B_max = {w: deque(maxlen=w) for w in SURPRISE_WINDOWS}
        self.sum_F = {w: 0.0 for w in ALL_WINDOWS}
        self.sum_B = {w: 0.0 for w in ALL_WINDOWS}
        self.all_F = self.all_B = 0.0
        self.last_F = self.last_B = 0.0
        self.d_cnt: dict = {}
        self.d_sum: dict = {}
        self.d_cards: dict = {}
        self.d_recent: dict = {}
        self.n_cnt: dict = {}
        self.n_sum: dict = {}
        self.c_cnt: dict = {}
        self.c_sum: dict = {}
        self.c_age: dict = {}
        self.day_hist: list[float] = []       # day of each processed row, indexed by position
        self.ratings: deque = deque(maxlen=RATING_WINDOW)   # previous ratings, user-level
        self.good_streak = 0.0                # consecutive ratings >= 3 ending at the last row
        self.lapse_streak = 0.0               # consecutive ratings == 1 ending at the last row
        self.prev_duration = 0.0              # previous row's response time (ms)

    # -- features of the next row -------------------------------------------
    def _window_features(self) -> dict[str, float]:
        t = self.t
        out = {
            "x16": self.sum_F[16] / (min(t, 16) + SHRINK),
            "x256": self.sum_F[256] / (min(t, 256) + SHRINK),
        }
        for w in SURPRISE_WINDOWS:
            den = max(1.0, float(min(t, w)))
            out[f"sF_mean{w}"] = self.sum_F[w] / den
            out[f"sB_mean{w}"] = self.sum_B[w] / den
            out[f"sF_max{w}"] = max(self.win_F_max[w]) if self.win_F_max[w] else 0.0
            out[f"sB_max{w}"] = max(self.win_B_max[w]) if self.win_B_max[w] else 0.0
        den_all = max(1.0, float(t))
        out["sF_last"], out["sB_last"] = self.last_F, self.last_B
        out["sF_all"], out["sB_all"] = self.all_F / den_all, self.all_B / den_all
        return out

    def _deck_features(self, card: int, deck: int, note: int, day: float) -> dict[str, float]:
        cnd, smd = self.d_cnt.get(deck, 0), self.d_sum.get(deck, 0.0)
        cnc, smc = self.c_cnt.get(card, 0), self.c_sum.get(card, 0.0)
        ntn, smn = self.n_cnt.get(note, 0), self.n_sum.get(note, 0.0)
        n_other, nn_other = cnd - cnc, ntn - cnc
        seen = self.d_recent.get(deck)
        last_other = -1 if seen is None else (seen[1] if seen[0] != card else seen[3])
        return {
            "st_card_seq": float(cnc),
            "st_age_days": self.c_age.get(card, 0.0),
            "dk_acc_other": (smd - smc) / (n_other + SHRINK),
            "dk_n_other": float(n_other),
            "dk_acc_all": smd / (cnd + SHRINK),
            "dk_n_all": float(cnd),
            "dk_cards_seen": float(self.d_cards.get(deck, 0)),
            "dk_since_other": (day - self.day_hist[last_other]) if last_other >= 0 else -1.0,
            "nt_acc_other": (smn - smc) / (nn_other + SHRINK),
            "nt_n_other": float(nn_other),
        }

    def _advance(self, card: int, deck: int, note: int, day: float, rB: float,
                 elapsed: float) -> None:
        seen = self.d_recent.get(deck)
        self.d_recent[deck] = ((card, self.t, seen[2], seen[3]) if seen and seen[0] == card
                               else (card, self.t, seen[0], seen[1]) if seen else
                               (card, self.t, -1, -1))
        cnd, cnc = self.d_cnt.get(deck, 0), self.c_cnt.get(card, 0)
        if cnd == 0:
            self.d_cards[deck] = 1
        elif cnc == 0:
            self.d_cards[deck] = self.d_cards.get(deck, 0) + 1
        self.d_cnt[deck] = cnd + 1
        self.d_sum[deck] = self.d_sum.get(deck, 0.0) + rB
        self.c_cnt[card] = cnc + 1
        self.c_sum[card] = self.c_sum.get(card, 0.0) + rB
        self.c_age[card] = self.c_age.get(card, 0.0) + float(elapsed)
        self.n_cnt[note] = self.n_cnt.get(note, 0) + 1
        self.n_sum[note] = self.n_sum.get(note, 0.0) + rB
        self.day_hist.append(float(day))

    def _stream_features(self, rating: float, elapsed_seconds: float) -> dict[str, float]:
        """Rating history + sub-day timing, exactly as `stage2.py` computes them.

        `rating` is this review's button; the windows use only *previous* reviews (0.0 for
        the first row), matching `_roll_mean_prev` / `_streak_prev`.
        """
        r = list(self.ratings)
        n8, n32 = r[-8:], r
        mean32 = float(np.mean(n32)) if n32 else 0.0
        var32 = float(np.mean(np.square(n32))) - mean32 ** 2 if n32 else 0.0
        return {
            "rt_mean8": float(np.mean(n8)) if n8 else 0.0,
            "rt_mean32": mean32,
            "rt_sd32": float(np.sqrt(max(0.0, var32))),
            "rt_last": r[-1] if r else 0.0,
            "rt_fail32": float(np.mean([x < 3 for x in n32])) if n32 else 0.0,
            "rt_good_streak": self.good_streak,
            "rt_lapse_streak": self.lapse_streak,
            "sg_log_secs": float(np.log1p(max(float(elapsed_seconds), 0.0))),
            "sg_same_session": 1.0 if float(elapsed_seconds) < SAME_SESSION_SECONDS else 0.0,
            "sg_prev_dur_log": float(np.log1p(self.prev_duration)),
        }

    # -- public API ----------------------------------------------------------
    def observe(self, *, y: float, p_fsrs: float, i: int, elapsed_days: float, lapse: int,
                nth_today: int, day: float, card_id: int, deck_id: int, note_id: int,
                p_b: float | None = None, rating: float | None = None,
                elapsed_seconds: float | None = None, duration: float | None = None) -> float:
        """Predict this review from earlier rows, then fold the review into the state.

        `rating`/`elapsed_seconds`/`duration` are required by models that carry the `stream`
        family and rejected when missing -- a serving path without them must not guess.
        """
        if self.stream and (rating is None or elapsed_seconds is None or duration is None):
            missing = [n for n, v in (("rating", rating), ("elapsed_seconds", elapsed_seconds),
                                      ("duration", duration)) if v is None]
            raise ValueError(f"model needs stream inputs; missing {missing}")
        feats = self._window_features()
        if p_b is None:
            if self.beta is None:
                raise ValueError("p_b not supplied and no per-user beta registered")
            z = _logit(p_fsrs)
            p_b = 1.0 / (1.0 + np.exp(-(z + self.beta[0] + self.beta[1] * feats["x16"]
                                       + self.beta[2] * feats["x256"])))
        p_b = float(p_b)
        if not 0 <= p_b <= 1:
            raise ValueError("p_b must be a finite probability in [0, 1]")
        # Clip only the logit, not residual history shared with batch feature building.
        feats.update(self._deck_features(int(card_id), int(deck_id), int(note_id), float(day)))
        feats.update({
            "st_i": float(i), "st_elapsed_days": float(elapsed_days), "st_lapse": float(lapse),
            "st_t": float(self.t), "st_nth_today": float(nth_today),
            "st_log_elapsed": float(np.log1p(elapsed_days)), "st_log_i": float(np.log1p(i)),
            "st_log_t": float(np.log1p(self.t)),
        })
        feats.update(self._stream_features(0.0 if rating is None else rating,
                                           0.0 if elapsed_seconds is None else elapsed_seconds))
        missing = set(self.names) - set(feats)
        if missing:
            raise KeyError(f"feature build is incomplete: {sorted(missing)}")
        self.last_features = feats
        X = np.array([[feats[n] for n in self.names]], dtype=np.float32)
        dm = xgb.DMatrix(X, base_margin=np.array([_logit(p_b)], dtype=np.float32),
                         missing=np.nan, feature_names=self.names)
        margin = float(self.bst.predict(dm, output_margin=True,
                                       iteration_range=self.iteration_range)[0])
        p_hat = 1.0 / (1.0 + np.exp(-margin))

        rF, rB = float(y) - float(p_fsrs), float(y) - p_b
        for w in ALL_WINDOWS:
            ev_F = self.win_F[w][0] if len(self.win_F[w]) == w else 0.0
            ev_B = self.win_B[w][0] if len(self.win_B[w]) == w else 0.0
            self.win_F[w].append(rF)
            self.win_B[w].append(rB)
            self.sum_F[w] += rF - ev_F
            self.sum_B[w] += rB - ev_B
        for w in SURPRISE_WINDOWS:
            self.win_F_max[w].append(rF)
            self.win_B_max[w].append(rB)
        self.all_F += rF
        self.all_B += rB
        self.last_F, self.last_B = rF, rB
        self._advance(int(card_id), int(deck_id), int(note_id), float(day), rB, elapsed_days)
        if rating is not None:
            r = float(rating)
            self.ratings.append(r)
            self.good_streak = self.good_streak + 1.0 if r >= 3 else 0.0
            self.lapse_streak = self.lapse_streak + 1.0 if r == 1 else 0.0
        if duration is not None:
            self.prev_duration = float(np.nan_to_num(duration, nan=0.0))
        self.t += 1
        return p_hat


# ---------------------------------------------------------------- self-check
def verify(users: list[int], model: str | Path, panel: str = "lockbox",
           max_rows: int | None = None, stream_table: Path | None = None,
           pred_file: Path | None = None, pred_col: str = "p_all") -> dict:
    """Replay stored rows through the stream and diff features + predictions.

    Models without the `stream` family diff against the probe's own feature table.  Models
    with it additionally need `stream_table` (the stage-2 per-row table holding the ten
    derived features) and read `rating` / `elapsed_seconds` / `duration` from the revlog
    frames, which is where `stage2.py` got them; the prediction reference is `pred_file`
    (default: the probe's `preds-*.parquet`, column `p_all`).
    """
    scorer = StreamScorer(model, panel=panel)
    stream = scorer.stream
    if stream and stream_table is None:
        raise ValueError("this model needs the stream family: pass --stream-table")
    names = [n for n in scorer.names if n not in STREAM_FEATS] if stream else list(scorer.names)
    parquet = HERE / ("features-all.parquet" if panel == "lockbox" else "features-fresh.parquet")
    preds = Path(pred_file) if pred_file else HERE / (
        "preds-eval.parquet" if panel == "lockbox" else "preds-fresh.parquet")
    have = set(pq.ParquetFile(parquet).schema_arrow.names)
    cols = list(dict.fromkeys(c for c in INPUT_COLUMNS + names if c in have))
    has_preds = preds.exists() and pred_col in set(pq.ParquetFile(preds).schema_arrow.names)
    frame_reader = None
    if stream:
        sys_path = str(HERE)
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import probe  # heavy import, only for stream models
        frame_reader = probe.scored_frame
    observe_seconds = 0.0
    observe_calls = 0
    summary = {"model": str(model), "panel": panel, "users": [], "feature_max_abs": {},
               "pred_max_abs": 0.0, "pred_rows": 0, "pred_file": str(preds),
               "pred_col": pred_col, "stream": stream}
    for user in users:
        rows = pq.read_table(parquet, columns=cols, filters=[("user", "==", int(user))])
        d = {c: rows.column(c).to_numpy(zero_copy_only=False) for c in rows.column_names}
        order = np.argsort(d["row_index"], kind="stable")
        if max_rows:
            order = order[:max_rows]
        if stream:
            assert np.array_equal(order, np.arange(len(order))), \
                f"user {user}: row_index not ascending, ordering assumption broken"
            st = pq.read_table(Path(stream_table), columns=["user", "row_index"] + STREAM_FEATS,
                               filters=[("user", "==", int(user))])
            for f in STREAM_FEATS:
                d[f] = st.column(f).to_numpy(zero_copy_only=False)
            assert np.array_equal(st.column("row_index").to_numpy(zero_copy_only=False),
                                  d["row_index"]), f"user {user}: stream table misaligned"
            frame = frame_reader(int(user))
            assert len(frame) == len(d["y"]), f"user {user}: frame {len(frame)} vs table"
            d["rating"] = frame["rating"].to_numpy(np.float64)
            d["elapsed_seconds"] = frame["elapsed_seconds"].to_numpy(np.float64)
            d["duration"] = frame["duration"].to_numpy(np.float64)
        stored = None
        if has_preds:
            pr = pq.read_table(preds, columns=["row_index", pred_col],
                               filters=[("user", "==", int(user))])
            by_index = dict(zip(pr.column("row_index").to_numpy().tolist(),
                                pr.column(pred_col).to_numpy().tolist()))
            idx = [int(d["row_index"][r]) for r in order]
            if idx and all(i in by_index for i in idx):
                stored = np.array([by_index[i] for i in idx], np.float64)
            elif idx:
                print(f"user {user:>5}: not in {preds.name} (features only)")
        scorer.begin_user()
        got = {n: np.zeros(len(order), np.float64) for n in scorer.names}
        p_hat = np.zeros(len(order), np.float64)
        for k, r in enumerate(order):
            t0 = time.perf_counter()
            p_hat[k] = scorer.observe(
                y=float(d["y"][r]), p_fsrs=float(d["p_fsrs"][r]), i=int(d["i_raw"][r]),
                elapsed_days=float(d["elapsed_days"][r]), lapse=int(d["lapse"][r]),
                nth_today=int(d["st_nth_today"][r]), day=float(d["day"][r]),
                card_id=int(d["card_id"][r]), deck_id=int(d["deck_id"][r]),
                note_id=int(d["note_id"][r]), p_b=float(d["p_b"][r]),
                rating=float(d["rating"][r]) if stream else None,
                elapsed_seconds=float(d["elapsed_seconds"][r]) if stream else None,
                duration=float(d["duration"][r]) if stream else None)
            observe_seconds += time.perf_counter() - t0
            observe_calls += 1
            for n in scorer.names:
                got[n][k] = scorer.last_features[n]
        for n in scorer.names:
            delta = float(np.abs(got[n] - d[n][order].astype(np.float64)).max())
            summary["feature_max_abs"][n] = max(summary["feature_max_abs"].get(n, 0.0), delta)
        entry = {"user": int(user), "rows": int(len(order)),
                 "feature_max_abs_overall": float(max(
                     float(np.abs(got[n] - d[n][order].astype(np.float64)).max())
                     for n in scorer.names))}
        if stored is not None:
            entry["pred_max_abs"] = float(np.abs(p_hat - stored).max())
            summary["pred_max_abs"] = max(summary["pred_max_abs"], entry["pred_max_abs"])
            summary["pred_rows"] += len(order)
        summary["users"].append(entry)
        print(f"user {user:>5}: {len(order):>6} rows  max|Δfeature| "
              f"{entry['feature_max_abs_overall']:.3e}"
              + (f"  max|Δp| {entry['pred_max_abs']:.3e}" if stored is not None else ""))
    summary["observe_us_per_review"] = 1e6 * observe_seconds / max(1, observe_calls)
    summary["observe_calls"] = observe_calls
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["verify"])
    ap.add_argument("--users", type=int, nargs="+", default=[16, 21, 32])
    ap.add_argument("--model", default=str(HERE / "model-all.json"))
    ap.add_argument("--panel", default="lockbox", choices=["lockbox", "fresh"])
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--stream-table", type=Path, default=None,
                    help="stage-2 per-row table with the ten stream features (stream models)")
    ap.add_argument("--pred-file", type=Path, default=None,
                    help="per-row reference predictions (default: the probe's preds-*.parquet)")
    ap.add_argument("--pred-col", default="p_all", help="reference column in --pred-file")
    args = ap.parse_args()
    out = verify(args.users, args.model, args.panel, args.max_rows, args.stream_table,
                 args.pred_file, args.pred_col)
    worst = max(out["feature_max_abs"].items(), key=lambda kv: kv[1])
    print(f"\nworst feature: {worst[0]} {worst[1]:.3e}")
    print(f"prediction max|Δp|: {out['pred_max_abs']:.3e} over {out['pred_rows']} rows")
    print(f"serving cost: {out['observe_us_per_review']:.1f} µs/review "
          f"({out['observe_calls']} reviews, feature build + single-row predict)")


if __name__ == "__main__":
    main()
