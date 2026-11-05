"""Team bear: relentless aggression.

Estimates equity against a random hand by Monte Carlo, then applies pressure:
wide preflop raises, frequent large bets, many bluffs. It tracks how often the
opponent folds to our bets and backs off the bluffing when they never fold.
"""

from __future__ import annotations

import itertools
import random
import time

from player_sdk.bot_interface import PokerBot
from player_sdk.models import Action, GameState

RANKS = "23456789TJQKA"
SUITS = "SHDC"
DECK = [r + s for r in RANKS for s in SUITS]
RANK_OF = {r: i + 2 for i, r in enumerate(RANKS)}


def _score5(cards):
    """Comparable tuple for a 5-card hand (higher is better)."""
    vals = sorted((RANK_OF[c[0]] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(vals), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_high = 5
    groups = sorted(((vals.count(v), v) for v in uniq), reverse=True)
    shape = [g[0] for g in groups]
    order = [g[1] for g in groups]
    if straight_high and flush:
        return (8, straight_high)
    if shape[0] == 4:
        return (7, *order)
    if shape == [3, 2]:
        return (6, *order)
    if flush:
        return (5, *vals)
    if straight_high:
        return (4, straight_high)
    if shape[0] == 3:
        return (3, *order)
    if shape == [2, 2, 1]:
        return (2, *order)
    if shape[0] == 2:
        return (1, *order)
    return (0, *vals)


def _best(cards):
    if len(cards) == 5:
        return _score5(cards)
    return max(_score5(c) for c in itertools.combinations(cards, 5))


def equity(hole, board, samples, rng):
    """Win probability (ties count half) against one random opponent hand."""
    known = set(hole) | set(board)
    rest = [c for c in DECK if c not in known]
    need = 5 - len(board)
    wins = 0.0
    for _ in range(samples):
        draw = rng.sample(rest, need + 2)
        opp = draw[:2]
        full = list(board) + draw[2:]
        mine = _best(list(hole) + full)
        theirs = _best(opp + full)
        if mine > theirs:
            wins += 1
        elif mine == theirs:
            wins += 0.5
    return wins / samples


class BearBot(PokerBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(2024)
        self.hand_key = None
        self.pending = None  # street of our last raise, awaiting a response
        self.bets_made = 0
        self.bets_folded = 0
        self.bets_called = 0

    # ---- opponent model -------------------------------------------------
    def _track(self, gs, key):
        if key != self.hand_key:
            if self.pending is not None:
                self.bets_made += 1
                self.bets_folded += 1  # hand ended with no reply to our bet
            self.pending = None
            self.hand_key = key
        elif self.pending is not None:
            self.bets_made += 1
            if gs.street != self.pending or gs.call_amount > 0:
                self.bets_called += 1
            self.pending = None

    def _fold_rate(self):
        if self.bets_made < 15:
            return 0.4
        return self.bets_folded / self.bets_made

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _raise(gs, target):
        low = gs.min_raise_to or 1
        high = gs.max_raise_to or low
        return Action.raise_to(int(max(min(target, high), min(low, high))))

    def act(self, gs: GameState) -> Action:
        t0 = time.time()
        board = gs.community_cards
        key = gs.hole_cards  # new hole cards mean a new hand
        self._track(gs, key)

        samples = 100 if len(board) < 5 else 140
        eq = equity(gs.hole_cards, board, samples, self.rng)

        can_raise = gs.can_raise or gs.can_bet
        to_call = gs.call_amount
        pot = gs.pot_size
        passive_opp = self._fold_rate() < 0.22  # they call us down: stop bluffing
        r = self.rng.random()
        bluff_freq = 0.25 if passive_opp else 0.6

        # ---- facing no bet: check or bet -------------------------------
        if to_call == 0:
            if not can_raise:
                return Action.call()
            if eq >= 0.70:
                return self._go(gs, 1.0 * pot)
            if eq >= 0.52:
                return self._go(gs, 0.75 * pot)
            if eq >= 0.38 and r < (0.85 if not passive_opp else 0.5):
                return self._go(gs, 0.66 * pot)
            if r < bluff_freq:
                return self._go(gs, 0.7 * pot)
            return Action.call()

        # ---- facing a bet ----------------------------------------------
        price = to_call / (pot + to_call) if pot + to_call else 0.0
        big = to_call > 0.6 * max(pot - to_call, 1)
        preflop = not board

        if can_raise:
            if eq >= 0.72:
                return self._go(gs, 1.0 * pot + to_call)
            if preflop and eq >= 0.45 and to_call <= 3 * (gs.big_blind or 20) and r < 0.6:
                return self._go(gs, 1.0 * pot + to_call)
            if not big and eq >= 0.5 and r < 0.5:
                return self._go(gs, 0.8 * pot + to_call)
            if not big and not passive_opp and eq >= 0.3 and r < 0.15:
                return self._go(gs, 0.9 * pot + to_call)  # bluff raise

        if eq >= price + 0.04:
            return Action.call()
        if preflop and to_call <= (gs.big_blind or 20) and eq >= 0.38:
            return Action.call()
        return Action.fold()

    def _go(self, gs, amount):
        self.pending = gs.street
        return self._raise(gs, (gs.my_bet or 0) + int(amount))
