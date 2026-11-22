#!/usr/bin/env python
"""Sweep diagnostic - characterise fallback rates and action distributions.

General-purpose (not milestone-specific): reads all *.parquet files under any
sweep's data directory and prints:
  1. Fallback rate by (game, opponent) - all games
  2. P(action | own_card) for Leduc LM rows - Leduc-specific card-blind test
     (skips cleanly when no Leduc rows are present)
  3. Leduc fallback rate by node type (facing-bet vs free) - Leduc LM rows only
  4. Mean LM reward by (game, opponent) - all games

TODO: add game-specific sections for Liar's Dice (dice-conditional action
distribution) and Dark Hex (probing behaviour), mirroring the Leduc section.

Usage:
    uv run python scripts/diagnose.py data/m5_pass1/
    uv run python scripts/diagnose.py            # default: data/m4_pass1/

Output is intended to be kept as a dated record of the sweep.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)


def _extract_card(info_str: str) -> int | None:
    m = re.search(r"\[Private:\s*(\d+)\]", info_str)
    return int(m.group(1)) if m else None


def _is_facing_bet(info_str: str) -> bool:
    """True if the info-state string indicates a facing-bet node.

    Leduc info-states use space-separated integer action codes in the
    [RoundN: ...] field.  Action 2 = Bet/Raise; if the last action in
    the current round was 2, the current player is facing a bet/raise.
    """
    # Find the current round from [Round N] field, then read that round's seq.
    round_m = re.search(r"\[Round\s*(\d+)\]", info_str)
    current_round = int(round_m.group(1)) if round_m else 1

    seq_m = re.search(r"\[Round" + str(current_round) + r":\s*([^\]]*)\]", info_str)
    if seq_m:
        seq = seq_m.group(1).strip()
        parts = seq.split()
        # If the last action in this round was action 2 (Bet/Raise), we're facing it.
        return bool(parts) and parts[-1] == "2"

    # Kuhn: simple info-set string ending in 'b'
    if info_str.endswith("b"):
        return True
    return False


def main(data_dir: Path) -> int:
    try:
        import polars as pl
    except ImportError:
        print("ERROR: polars required.", file=sys.stderr)
        return 1

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: No parquet files in {data_dir}", file=sys.stderr)
        return 1

    print(f"Loading {len(files)} Parquet files from {data_dir}/\n")
    dfs = []
    unreadable = []
    for f in files:
        try:
            dfs.append(pl.read_parquet(f))
        except Exception as e:
            unreadable.append(f"{f}: {e}")
    if unreadable:
        # A shard that cannot be read is missing from every rate below, and run_experiment acts
        # only on a non-zero exit, so it must not be a warning.
        print(
            f"ERROR: {len(unreadable)} shard(s) could not be read; nothing is reported:\n  "
            + "\n  ".join(unreadable),
            file=sys.stderr,
        )
        return 1
    # Schema fingerprinting at lmopt.analysis.shards.load_frames.
    from lmopt.analysis.shards import load_frames

    df = load_frames(dfs, names=[str(f) for f in files])
    lm = df.filter(pl.col("prompt").is_not_null())
    print(f"Total LM decision nodes: {len(lm):,}")
    print(f"Games: {sorted(lm['game'].unique().to_list())}")

    # ------------------------------------------------------------------ #
    # 1. Fallback rate by (game, opponent) - overall
    # ------------------------------------------------------------------ #
    print("\n=== 1. Fallback rate by (game, opponent) ===")
    print(
        f"{'game':<12} {'opponent':<14} {'nodes':>7} {'fallbacks':>10} {'fb_rate':>8}"
    )
    print("-" * 57)

    _OPP_ALIASES = {
        "cfr_ref": "cfr_ref",
        "cfrref": "cfr_ref",
        "bayesian": "bayesian",
        "always_bet": "always_bet",
        "alwaysbet": "always_bet",
        "rl": "rl",
    }

    def _opp_from_matchup(mid: str) -> str:
        for alias, canonical in _OPP_ALIASES.items():
            if alias in mid:
                return canonical
        return mid

    lm2 = lm.with_columns(
        pl.col("matchup_id")
        .map_elements(_opp_from_matchup, return_dtype=pl.Utf8)
        .alias("opponent")
    )
    fb_summary = (
        lm2.group_by(["game", "opponent"])
        .agg(
            pl.len().alias("nodes"),
            pl.col("fallback_used").sum().alias("fallbacks"),
        )
        .with_columns((pl.col("fallbacks") / pl.col("nodes")).alias("fb_rate"))
        .sort(["game", "opponent"])
    )
    for row in fb_summary.iter_rows(named=True):
        print(
            f"{row['game']:<12} {row['opponent']:<14} {row['nodes']:>7} "
            f"{row['fallbacks']:>10} {row['fb_rate']:>8.3f}"
        )

    # ------------------------------------------------------------------ #
    # 2. Leduc: P(action | own_card) - card-blind hypothesis test
    # ------------------------------------------------------------------ #
    print("\n=== 2. Leduc action distribution by own_card ===")
    leduc_lm = lm2.filter(pl.col("game") == "leduc")
    if leduc_lm.is_empty():
        print("  No Leduc LM rows found.")
    else:
        leduc_lm = leduc_lm.with_columns(
            pl.col("info_set")
            .map_elements(_extract_card, return_dtype=pl.Int32)
            .alias("own_card")
        )
        card_names = {0: "J1", 1: "J2", 2: "Q1", 3: "Q2", 4: "K1", 5: "K2"}
        action_names = {0: "Fold", 1: "Call/Check", 2: "Bet/Raise"}
        dist = (
            leduc_lm.group_by(["own_card", "parsed_action"])
            .agg(pl.len().alias("n"))
            .sort(["own_card", "parsed_action"])
        )
        print(f"  {'card':<6} {'action':<12} {'n':>6}")
        for row in dist.iter_rows(named=True):
            card_lbl = card_names.get(row["own_card"], str(row["own_card"]))
            act_lbl = action_names.get(row["parsed_action"], str(row["parsed_action"]))
            print(f"  {card_lbl:<6} {act_lbl:<12} {row['n']:>6}")

        # Unique raw completions - confirms whether model is card-blind
        n_unique = leduc_lm["raw_completion"].n_unique()
        print(f"\n  Unique raw_completion values in Leduc: {n_unique}")
        if n_unique <= 5:
            for v in leduc_lm["raw_completion"].unique().to_list():
                print(f"    {repr(v)}")

    # ------------------------------------------------------------------ #
    # 3. Fallback by node type (facing-bet vs free)
    # ------------------------------------------------------------------ #
    print("\n=== 3. Leduc fallback rate by node type (facing-bet vs free) ===")
    if leduc_lm.is_empty():
        print("  No Leduc LM rows found.")
    else:
        leduc_lm2 = leduc_lm.with_columns(
            pl.col("info_set")
            .map_elements(_is_facing_bet, return_dtype=pl.Boolean)
            .alias("facing_bet")
        )
        fb_by_type = (
            leduc_lm2.group_by("facing_bet")
            .agg(
                pl.len().alias("nodes"),
                pl.col("fallback_used").sum().alias("fallbacks"),
            )
            .with_columns((pl.col("fallbacks") / pl.col("nodes")).alias("fb_rate"))
            .sort("facing_bet")
        )
        print(f"  {'facing_bet':<12} {'nodes':>7} {'fallbacks':>10} {'fb_rate':>8}")
        for row in fb_by_type.iter_rows(named=True):
            print(
                f"  {str(row['facing_bet']):<12} {row['nodes']:>7} "
                f"{row['fallbacks']:>10} {row['fb_rate']:>8.3f}"
            )

    # ------------------------------------------------------------------ #
    # 4. Mean reward by (game, opponent)
    # ------------------------------------------------------------------ #
    print("\n=== 4. Mean LM reward by (game, opponent) ===")
    print(f"{'game':<12} {'opponent':<14} {'mean_reward':>12} {'seeds':>6}")
    print("-" * 48)
    reward_summary = (
        lm2.group_by(["game", "opponent", "matchup_id", "seed", "hand_idx"])
        .agg(pl.col("reward").first())
        .group_by(["game", "opponent"])
        .agg(
            pl.col("reward").mean().alias("mean_reward"),
            pl.col("seed").n_unique().alias("n_seeds"),
        )
        .sort(["game", "opponent"])
    )
    for row in reward_summary.iter_rows(named=True):
        print(
            f"{row['game']:<12} {row['opponent']:<14} {row['mean_reward']:>12.4f} "
            f"{row['n_seeds']:>6}"
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/m4_pass1")
    sys.exit(main(data_path))
