"""Arm-collapse detection: when two arms are one observed policy, say so.

Two different models can produce
byte-identical play, and every downstream cross-model table then reads as two
models independently agreeing when the truth is one policy observed twice.
That is a materially stronger claim than the data supports.

On the
v4 keepable arms OLMo-2-13B and Mistral-Nemo are equal on the full play key for
Leduc (both 37,765 LM rows, two distinct completions each at identical
counts, identical reward totals) and are not equal for Kuhn (row counts
differ outright). The v2 arms collapsed on both. So collapse is a property of
the data that appears and vanishes per epoch, which is exactly why this is a
standing check and not a one-off sweep or a hardcoded model pair.

Design, and why it is split in two. Collapse is inherently cross-arm: it takes
two arms to see. But `cross_arm.py` reads analysis artifacts and never raw
Parquet, by deliberate design. So the work splits along that existing boundary:

- `policy_fingerprint` is a PER-ARM quantity computed where the Parquet is
  already open, and travels with the arm's artifacts.
- `detect_collapse` is a cross-ARM comparison over those fingerprints, needing
  no Parquet at all.

The alternative, re-reading every arm's shards in the cross-arm layer, would
invert that boundary and made every future cross-arm consumer pay the
Parquet cost to ask a one-line question.

What this module deliberately does not do: merge, drop, deduplicate or
suppress a collapsed arm. The arms are real and separately collected; the
defect is in how they are read. Labelling is the fix. Suppression would destroy
the evidence that two models converged, which is itself a finding.
"""

from __future__ import annotations
import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
import polars as pl

# The columns that define "the same play": every logged decision field
# except `reward` and `matchup_id`. Reward is a
# function of the actions and including it would make the fingerprint
# redundant, while matchup_id embeds the arm's own identity in some layouts
# and would prevent any two arms from ever matching.
PLAY_KEY: tuple[str, ...] = (
    "seed",
    "hand_idx",
    "node_idx",
    "player_to_act",
    "info_set",
    "parsed_action",
)


class FingerprintUnavailable(ValueError):
    """Raised when a frame cannot be fingerprinted.

    Deliberately an exception rather than a sentinel return: a silently
    absent fingerprint would make two arms look non-collapsed, which is the
    permissive direction and reintroduces the defect this module exists to
    catch (no silent fallbacks).
    """


def policy_fingerprint(df: pl.DataFrame, *, key: Sequence[str] = PLAY_KEY) -> str:
    """
    Return a stable sha256 over one arm's play on one game.

    Deterministic across row order and across processes: rows are sorted by
    the key before hashing, and the digest is taken over a canonical
    CSV rendering rather than over Python object hashes (`hash()` is salted
    per process and would make fingerprints incomparable between runs).
    :param df: one arm's rows for one game. Passing more than one game silently
        produces a fingerprint of the mixture, so callers group first.
    :param key: columns defining "the same play"; defaults to `PLAY_KEY`.
    :returns: Hex sha256 digest.
    :raises FingerprintUnavailable: if any key column is missing, or the frame is
        empty. Both are refusals rather than defaults, because an empty
        frame would otherwise hash to a constant and make every empty arm
        collapse with every other.
    """
    missing = [c for c in key if c not in df.columns]
    if missing:
        raise FingerprintUnavailable(
            f"cannot fingerprint: missing key column(s) {missing}; "
            f"frame has {sorted(df.columns)}"
        )
    if df.height == 0:
        raise FingerprintUnavailable(
            "cannot fingerprint an empty frame: every empty frame would hash "
            "alike and be reported as collapsed with every other"
        )
    ordered = df.select(list(key)).sort(list(key))
    payload = ordered.write_csv().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


#: The columns the first-listed-legal-action share is computed from.
FIRST_LISTED_KEY: tuple[str, ...] = ("legal_mask", "parsed_action")


def first_listed_action_share(df: pl.DataFrame) -> tuple[float, int]:
    """
    how often one arm chose the first legal action the prompt lists, on one game
    A collapse between two arms is usually not a coincidence: both arms answer with the
    first option, so the share belongs beside the collapse marker as the reading of it.
    :param df: one arm's rows for one game, the scored LM's and the opponent's together
    :returns: (share of the arm's kept LM rows whose parsed action is the lowest-index legal
        action, the number of those rows)
    :raises FingerprintUnavailable: when a needed column is absent or no LM row survives the
        fallback drop, since a defaulted share would read as a measured one
    """
    from lmopt.scoring._actor import _lm_mask, drop_fallback_rows

    missing = [c for c in FIRST_LISTED_KEY if c not in df.columns]
    if missing:
        raise FingerprintUnavailable(
            f"cannot read the first-listed share: missing column(s) {missing}"
        )
    kept = drop_fallback_rows(df).kept
    rows = kept.filter(
        _lm_mask(kept)
        & pl.col("parsed_action").is_not_null()
        & pl.col("legal_mask").is_not_null()
    )
    if rows.height == 0:
        raise FingerprintUnavailable(
            "cannot read the first-listed share: no LM row survives the fallback drop"
        )
    hits = rows.select(
        (pl.col("legal_mask").list.arg_max() == pl.col("parsed_action")).alias("hit")
    )["hit"]
    return float(hits.sum() / rows.height), rows.height


#: A constant policy needs at least this many distinct info-sets: one action at a single info-set is
#: not a policy that ignores the state, it is just one observation.
CONSTANT_MIN_INFOSETS = 2


@dataclasses.dataclass(frozen=True)
class ConstantAction:
    """
    the one action an arm played at every info-set it was scored on
    :param action: the parsed action index, the same at every kept LM row (int)
    :param label: the action's name when every row carries the same one, else None (str | None)
    :param n_infosets: the distinct info-sets the action was played at (int)
    :param n_rows: the kept LM rows behind it (int)
    """

    action: int
    label: str | None
    n_infosets: int
    n_rows: int

    def words(self) -> str:
        """
        :returns: the action as a reader sees it, the index with its name when it has one
        """
        return f"{self.action} ({self.label})" if self.label else str(self.action)


def constant_action(
    df: pl.DataFrame, actor: str | None = None
) -> ConstantAction | None:
    """
    the single action an arm played at every info-set of one game, or None when it played more
    A model that answers the same action whatever the state is one policy, not a policy that
    happens to agree with another model's: a distance computed from it has no seed-to-seed
    spread and its interval is zero-width for that reason alone.
    :param df: one arm's rows for one game
    :param actor: which agent's rows are the scored LM's, or None for the rows that carry a prompt
    :returns: the `ConstantAction` when the scored LM's kept rows hold one distinct parsed
        action across at least `CONSTANT_MIN_INFOSETS` info-sets; None when they hold two or more
        actions, or fewer info-sets than that. Rows with a single legal action are not counted
        when the frame records the legal mask, since a forced move is not a choice
    :raises FingerprintUnavailable: when a needed column is absent or no LM row survives the
        fallback drop, since None would then read as "played more than one action"
    """
    from lmopt.scoring._actor import _lm_mask, drop_fallback_rows

    missing = [c for c in ("info_set", "parsed_action") if c not in df.columns]
    if missing:
        raise FingerprintUnavailable(
            f"cannot read the constant action: missing column(s) {missing}"
        )
    if actor is not None:
        if "actor" not in df.columns:
            raise FingerprintUnavailable(
                f"cannot read the constant action of actor {actor!r}: no actor column"
            )
        df = df.filter(pl.col("actor") == actor)
    kept = drop_fallback_rows(df).kept
    rows = kept.filter(_lm_mask(kept) & pl.col("parsed_action").is_not_null())
    if rows.height == 0:
        raise FingerprintUnavailable(
            "cannot read the constant action: no LM row survives the fallback drop"
        )
    if "legal_mask" in rows.columns:
        rows = rows.filter(pl.col("legal_mask").list.sum() > 1)
        if rows.height == 0:
            return None
    actions = rows["parsed_action"].unique()
    n_infosets = rows["info_set"].n_unique()
    if actions.len() != 1 or n_infosets < CONSTANT_MIN_INFOSETS:
        return None
    labels = (
        rows["action_str"].drop_nulls().unique()
        if "action_str" in rows.columns
        else None
    )
    label = str(labels[0]) if labels is not None and labels.len() == 1 else None
    return ConstantAction(int(actions[0]), label, n_infosets, rows.height)


def detect_collapse(
    fingerprints: Mapping[str, str],
) -> list[tuple[str, ...]]:
    """
    Group arms sharing a fingerprint, for one game.
    :param fingerprints: arm name -> fingerprint, for a single game.
    :returns: Sorted tuples of arm names that share a fingerprint, one tuple per
        collapsed group, only for groups of 2 or more. Empty list when every
        arm is distinct. Groups are sorted internally and between themselves,
        so the output is stable for tests and for diffing artifacts.
    """
    by_digest: dict[str, list[str]] = {}
    for arm, digest in fingerprints.items():
        by_digest.setdefault(digest, []).append(arm)
    return sorted(tuple(sorted(arms)) for arms in by_digest.values() if len(arms) > 1)


def collapse_label(group: Sequence[str], game: str) -> str:
    """The sentence a reader must see beside any number from a collapsed group.

    Phrased to block the specific misreading the defect produces. It states
    that the arms are ONE observed policy, and that the effective number of
    independent observations is one, because a reader who sees only "these
    arms are identical" may still average them as though that were two
    measurements agreeing.
    """
    names = ", ".join(sorted(group))
    return (
        f"COLLAPSED on {game}: {names} are byte-identical on the play key, so "
        f"they are ONE observed policy, not {len(group)} models independently "
        f"agreeing. Effective independent observations for any {game} "
        f"statistic over this set: 1, not {len(group)}."
    )
