#!/usr/bin/env python
"""LP cross-check: independent validation of CFR+ equilibrium via sequence-form LP.

Solves Kuhn poker with both CFR+ and OpenSpiel's sequence-form LP (Koller,
Megiddo & von Stengel, STOC 1994), then confirms the game values agree within
tolerance.  This is the second independent validation pathway for an imported
and validated equilibrium reference: exploitability->0 is the first check; LP
cross-check is the second.

Why this matters for P6 (subgame perfection):
  The CFR+ average policy is one Nash equilibrium; the LP finds another. At
  on-path nodes both must give the same game value (unique by minimax theorem).
  Off-path strategies may differ (equilibrium non-uniqueness), which is precisely
  why solve_cfr_tremble() is needed for P6 - but the on-path game value
  agreement is the provable invariant we can test here.

Solver note:
  OpenSpiel's sequence_form_lp defaults to ECOS, which is not installed.  This
  script patches LinearProgram.solve to use OSQP (included with cvxpy) with the
  matching OSQP solver kwargs.

Leduc note:
  Leduc LP with OSQP can be slow (minutes) due to the larger sequence-form
  matrix. Run with --all to include Leduc; the default covers Kuhn only, which
  is sufficient to validate the LP mechanism and the analytic game value (-1/18).

Usage:
    uv run python scripts/validate_lp.py          # Kuhn only (fast, ~1s)
    uv run python scripts/validate_lp.py --all    # Kuhn + Leduc (slow, minutes)
"""

from __future__ import annotations

import argparse
import math
import sys

import cvxpy as cp
from open_spiel.python.algorithms import expected_game_score, lp_solver as _lp
from open_spiel.python.algorithms import sequence_form_lp

from lmopt.games import load_kuhn, load_leduc
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

from lmopt.games.cfr import solve_cfr

# Patch LinearProgram.solve to use OSQP (installed) instead of ECOS (not installed).
# OSQP's solver kwargs are already defined in lp_solver.DEFAULT_OSQP_SOLVER_KWARGS.
_orig_solve = _lp.LinearProgram.solve


def _osqp_solve(
    self, solver=cp.OSQP, solver_kwargs=_lp.DEFAULT_OSQP_SOLVER_KWARGS, **kw
):
    return _orig_solve(self, solver=solver, solver_kwargs=solver_kwargs, **kw)


_lp.LinearProgram.solve = _osqp_solve

KUHN_ANALYTIC_VALUE = -1.0 / 18.0

# (name, loader, cfr_iters, value_tol, slow)
# value_tol: |v0_lp - v0_cfr| must be below this
# slow: only run with --all (Leduc LP can take minutes with OSQP)
SPECS = [
    ("kuhn", load_kuhn, 500, 1e-4, False),
    ("leduc", load_leduc, 1000, 1e-3, True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LP cross-check for CFR+ equilibrium.")
    p.add_argument(
        "--all", action="store_true", help="Include slow games (Leduc LP, minutes)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    specs = [s for s in SPECS if not s[4] or args.all]

    header = f"{'game':<8} {'v0_cfr':>10} {'v0_lp':>10} {'delta':>10} {'status':>8}"
    print(header)
    print("-" * len(header))

    failures: list[str] = []

    for name, loader, cfr_iters, val_tol, _slow in specs:
        game = loader()

        cfr_pol = solve_cfr(game, cfr_iters, plus=True)
        v0_cfr = expected_game_score.policy_value(
            game.new_initial_state(), [cfr_pol, cfr_pol]
        )[0]

        v0_lp, _v1_lp, _pol0, _pol1 = sequence_form_lp.solve_zero_sum_game(
            game, solver=cp.OSQP
        )

        # The LP cross-check is game-value agreement.  The minimax theorem
        # guarantees the game value is unique; if CFR+ (which converges to Nash)
        # and LP (which computes Nash exactly) agree on v0, the equilibrium
        # reference is independently validated.  Exploitability of the LP policy
        # itself is not checked here because solve_zero_sum_game returns separate
        # P0/P1 TabularPolicy objects and the joint exploitability API expects a
        # unified policy covering both players.
        delta = abs(v0_cfr - v0_lp)
        # A NaN delta compares False both ways, so finiteness is part of agreeing.
        ok = math.isfinite(delta) and delta < val_tol
        status = "OK" if ok else "FAIL"

        print(f"{name:<8} {v0_cfr:>+10.6f} {v0_lp:>+10.6f} {delta:>10.2e} {status:>8}")

        if not ok:
            failures.append(
                f"{name}: CFR/LP game-value delta {delta:.2e} >= tol {val_tol:.2e}"
            )

        if name == "kuhn":
            analytic_delta = abs(v0_lp - KUHN_ANALYTIC_VALUE)
            if not (math.isfinite(analytic_delta) and analytic_delta < 1e-4):
                failures.append(
                    f"kuhn: LP game value {v0_lp:.6f} departs from analytic "
                    f"{KUHN_ANALYTIC_VALUE:.6f} by {analytic_delta:.2e}"
                )

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("OK: CFR+ and sequence-form LP game values agree within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
