#!/usr/bin/env python
"""CFR convergence validation for all ten registry games (five canonical +
five depth variants).

Solves CFR+ for each game and prints a convergence table. Doubles as a CI gate:
exits non-zero if any game's exploitability exceeds its tolerance, or if the
Kuhn game value departs from the analytic equilibrium.

Run on the Linux box where OpenSpiel installs from a PyPI wheel:

    python scripts/validate_cfr.py
    python scripts/validate_cfr.py --game goofspiel_5 liarsdice_1d8
"""

from __future__ import annotations

import argparse
import math
import sys

from open_spiel.python.algorithms import expected_game_score

from lmopt.games import (
    load_darkhex,
    load_goofspiel,
    load_goofspiel_5,
    load_goofspiel_6,
    load_kuhn,
    load_kuhn_deep,
    load_leduc,
    load_leduc_deep,
    load_liarsdice,
    load_liarsdice_1d8,
)
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

from lmopt.games.cfr import exploitability_value, nash_conv, solve_cfr

# Analytic Kuhn poker game value to player 0 at equilibrium (Nash, first player
# is at a disadvantage): -1/18.
KUHN_GAME_VALUE = -1.0 / 18.0

# Per-game CFR budgets and exploitability tolerances. CFR+ converges quickly;
# Leduc has the largest canonical tree so it gets the most iterations.
# Depth-variant bars are set at reduced budgets from measured
# convergence on a Linux box, open_spiel 1.6.15 (re-measure if
# the pyspiel version changes):
#   goofspiel_5: expl 8.9e-4 @500 iters (~1.5 min)   bar 1e-2, margin 11x
# darkhex_2x3/darkhex_3x2 are not in this table and have no reference
# pickles. Use --game for a single-variant run.
#
# goofspiel_6: the CFR-tractability
# frontier; goofspiel_7 is out of reach
# (39.7GB RSS after 69+ min, no convergence). Tabular CFR+ is deterministic,
# so this margin is comfortable rather than statistical:
#   goofspiel_6: expl 8.9e-3 @200 iters (~20.6 min)   bar 1.5e-2, margin 1.7x
#
# liarsdice_1d8: the analogous
# CFR-tractability frontier for Liar's Dice; dice_sides=9 is out of reach (RSS
# 927MB->3.1GB in 12s):
#   liarsdice_1d8: expl 4.3e-3 @120 iters (~50.7 min)   bar 6e-3, margin 1.4x
#
# kuhn_deep: Kuhn poker's own depth axis,
# loaded via universal_poker. Unlike goofspiel_6/liarsdice_1d8 above,
# numRanks=13 is an engine STRUCTURAL ceiling (universal_poker segfaults
# above 13 ranks), not a CFR-tractability limit -- CFR+ converges trivially
# here, so the margin below is comfortable rather than thin by necessity:
#   kuhn_deep: expl 7.1e-5 @150 iters (~2.5s)   bar 2e-4, margin ~2.8x
#
# leduc_deep: Leduc's own depth axis, loaded
# via universal_poker, same numRanks=13 engine ceiling as kuhn_deep -- but
# Unlike kuhn_deep, CFR+ convergence here is not trivial: a ~1h43m
# timed solve is needed (Leduc's 2-round, raise-capable betting is far more
# expensive per iteration than Kuhn's single-round shape, even at the
# identical max deck size):
#   leduc_deep: expl 9.88e-4 @550 iters (~1h43m)   bar 1.5e-3, margin ~1.5x
#
# darkhex_3x3 is deliberately absent: it has no full CFR+
# reference and cannot have one - TabularPolicy's full-tree enumeration
# MemoryErrors (6 GB cap, 17.6 s), and solve_cfr would
# fare no better. This CI gate must never call solve_cfr on it; its reference
# is the partial MCCFR capture served via lmopt.games.depth_reference
# (lmopt.games.depth_reference.DEPTH_REFERENCES).
# (game key, loader, iterations, exploitability tolerance)
GAME_SPECS = [
    ("kuhn", load_kuhn, 300, 1e-3),
    ("leduc", load_leduc, 1000, 1e-2),
    ("goofspiel", load_goofspiel, 500, 1e-2),
    ("liarsdice", load_liarsdice, 500, 5e-3),
    ("darkhex", load_darkhex, 500, 1e-3),
    ("goofspiel_5", load_goofspiel_5, 500, 1e-2),
    ("goofspiel_6", load_goofspiel_6, 200, 1.5e-2),
    ("liarsdice_1d8", load_liarsdice_1d8, 120, 6e-3),
    ("kuhn_deep", load_kuhn_deep, 150, 2e-4),
    ("leduc_deep", load_leduc_deep, 550, 1.5e-3),
]


def _breach(value: float, limit: float) -> bool:
    """
    whether a measured value fails its limit: above it, or not a finite number at all, since a NaN
    compares False with everything and would otherwise pass the scientific-validity gate
    :param value: the measured quantity (float)
    :param limit: the largest value that passes (float)
    :returns: True when the value is not finite or exceeds the limit (bool)
    """
    return not math.isfinite(value) or value > limit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--game",
        nargs="+",
        choices=[name for name, *_ in GAME_SPECS],
        default=None,
        help="Validate only these games (default: all, the CI gate).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    specs = [s for s in GAME_SPECS if args.game is None or s[0] in args.game]

    header = (
        f"{'game':<12}{'iters':>8}{'nash_conv':>14}{'exploitability':>16}{'extra':>20}"
    )
    print(header)
    print("-" * len(header))

    failures: list[str] = []

    for name, loader, iters, tol in specs:
        game = loader()
        avg_policy = solve_cfr(game, iters, plus=True)
        nc = nash_conv(game, avg_policy)
        expl = exploitability_value(game, avg_policy)

        extra = ""
        if name == "kuhn":
            value_p0 = expected_game_score.policy_value(
                game.new_initial_state(), [avg_policy, avg_policy]
            )[0]
            extra = f"v0={value_p0:+.5f}"
            if _breach(abs(value_p0 - KUHN_GAME_VALUE), 1e-3):
                failures.append(
                    f"kuhn game value {value_p0:+.5f} != analytic {KUHN_GAME_VALUE:+.5f}"
                )

        print(f"{name:<12}{iters:>8}{nc:>14.6f}{expl:>16.6f}{extra:>20}")

        if _breach(expl, tol):
            failures.append(f"{name} exploitability {expl:.6f} is not within tol {tol}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("OK: all games converged within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
