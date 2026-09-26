#!/usr/bin/env python
"""Membership-safe streaming scorer.

The frozen serve.py is retained for published-run parity. This subclass fixes
deck/note state when a card changes group by tracking card history within each
deck and note instead of subtracting the card's global history.
"""

from __future__ import annotations

import serve as _frozen

feature_names = _frozen.feature_names
load_betas = _frozen.load_betas
STREAM_FEATS = _frozen.STREAM_FEATS


class StreamScorer(_frozen.StreamScorer):
    def begin_user(self, beta: list[float] | None = None) -> None:
        super().begin_user(beta)
        self.dc_cnt: dict[tuple[int, int], int] = {}
        self.dc_sum: dict[tuple[int, int], float] = {}
        self.nc_cnt: dict[tuple[int, int], int] = {}
        self.nc_sum: dict[tuple[int, int], float] = {}
        self.d_seen_cards: dict[int, set[int]] = {}

    def _deck_features(
        self, card: int, deck: int, note: int, day: float
    ) -> dict[str, float]:
        cnd = self.d_cnt.get(deck, 0)
        smd = self.d_sum.get(deck, 0.0)
        dkey = (deck, card)
        dcn = self.dc_cnt.get(dkey, 0)
        dcs = self.dc_sum.get(dkey, 0.0)

        ntn = self.n_cnt.get(note, 0)
        smn = self.n_sum.get(note, 0.0)
        nkey = (note, card)
        ncn = self.nc_cnt.get(nkey, 0)
        ncs = self.nc_sum.get(nkey, 0.0)

        seen = self.d_recent.get(deck)
        last_other = -1 if seen is None else (seen[1] if seen[0] != card else seen[3])
        d_other = cnd - dcn
        n_other = ntn - ncn
        if d_other < 0 or n_other < 0:
            raise RuntimeError("group-local history invariant violated")

        return {
            "st_card_seq": float(self.c_cnt.get(card, 0)),
            "st_age_days": self.c_age.get(card, 0.0),
            "dk_acc_other": (smd - dcs) / (d_other + _frozen.SHRINK),
            "dk_n_other": float(d_other),
            "dk_acc_all": smd / (cnd + _frozen.SHRINK),
            "dk_n_all": float(cnd),
            "dk_cards_seen": float(len(self.d_seen_cards.get(deck, set()))),
            "dk_since_other": (
                day - self.day_hist[last_other] if last_other >= 0 else -1.0
            ),
            "nt_acc_other": (smn - ncs) / (n_other + _frozen.SHRINK),
            "nt_n_other": float(n_other),
        }

    def _advance(
        self,
        card: int,
        deck: int,
        note: int,
        day: float,
        rB: float,
        elapsed: float,
    ) -> None:
        seen = self.d_recent.get(deck)
        self.d_recent[deck] = (
            (card, self.t, seen[2], seen[3])
            if seen and seen[0] == card
            else (card, self.t, seen[0], seen[1])
            if seen
            else (card, self.t, -1, -1)
        )

        dkey = (deck, card)
        nkey = (note, card)
        cnd = self.d_cnt.get(deck, 0)
        cnc = self.c_cnt.get(card, 0)
        ntn = self.n_cnt.get(note, 0)

        cards = self.d_seen_cards.setdefault(deck, set())
        cards.add(card)
        self.d_cards[deck] = len(cards)

        self.d_cnt[deck] = cnd + 1
        self.d_sum[deck] = self.d_sum.get(deck, 0.0) + rB
        self.dc_cnt[dkey] = self.dc_cnt.get(dkey, 0) + 1
        self.dc_sum[dkey] = self.dc_sum.get(dkey, 0.0) + rB

        self.c_cnt[card] = cnc + 1
        self.c_sum[card] = self.c_sum.get(card, 0.0) + rB
        self.c_age[card] = self.c_age.get(card, 0.0) + float(elapsed)

        self.n_cnt[note] = ntn + 1
        self.n_sum[note] = self.n_sum.get(note, 0.0) + rB
        self.nc_cnt[nkey] = self.nc_cnt.get(nkey, 0) + 1
        self.nc_sum[nkey] = self.nc_sum.get(nkey, 0.0) + rB

        self.day_hist.append(float(day))


__all__ = ["StreamScorer", "feature_names", "load_betas", "STREAM_FEATS"]
