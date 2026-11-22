"""Prompt serialisation for Dark Hex (parametric; default instance is 2×2).

Game: dark_hex(board_size=num_rows) - imperfect-information connection game.

Info-state encoding (empirically verified):
  Line 0: row 0 of board
  Line 1: row 1 of board
  ...
  Line num_rows-1: row num_rows-1 of board
  Line num_rows: total move count (sequential turns in the game tree)
  Line num_rows+1: space-separated "player_id,action" move history

Board symbol conventions (player-relative):
  P0 perspective: 'x' = own (black) stone, 'o' = discovered P1 (white) stone
  P1 perspective: 'o' = own (white) stone, 'x' = discovered P0 (black) stone
  '.' = unknown (possibly empty, possibly opponent-occupied)

Winning conditions:
  P0 (black): connect top row to bottom row (vertical path)
  P1 (white): connect left column to right column (horizontal path)

Collision semantics (OpenSpiel's retry variant):
  Playing on an opponent-occupied cell: stone not placed, opponent's cell
  revealed as 'x' or 'o', same player moves again. Revealed cells become
  illegal (shown as non-'.' in info state and excluded from legal_actions).

Legal actions inferred from info state: all cells showing '.' are legal.

Belief elicitation: per-cell probability the opponent occupies each unknown cell
(P8 board belief). Independent Bernoullis, each 0–1 (they do not sum to 1).
"""

from __future__ import annotations

import json
import re
import types
from typing import Optional

from lmopt.prompting.memory import MemoryTerms
from lmopt.prompting.spec import DARKHEX_SPEC as _SPEC
from lmopt.prompting.spec import (
    GamePromptSpec,
    finite_float,
    has_multiple_labels,
    in_display_order,
    labels_in_display_order,
    legal_response_suffix,
    load_belief_object,
    quoted_keys,
)


def parse_own_card(info_state: str) -> int | None:
    """No private card in Dark Hex - always returns None."""
    return None


def _column_letters(vocab, num_cols: int) -> list[str]:
    """The column labels this instance renders, left -> right.

    Canonical is A.. (``string.ascii_uppercase``). A variant spec may carry
    ``action_vocabulary["column_labels"]``, a sequence of single upper-case
    letters at least ``num_cols`` long, which is the P1 relabel injection point
    for this game (a column-label permutation). Validation is loud rather than defensive: a short or repeated
    alphabet would otherwise render a board whose labels collide or run off the
    end, which is a garbled prompt rather than a caught defect.
    """
    labels = vocab.get("column_labels")
    if labels is None:
        return [chr(ord("A") + c) for c in range(num_cols)]
    chosen = list(labels)[:num_cols]
    if len(chosen) < num_cols:
        raise ValueError(
            f"column_labels supplies {len(chosen)} labels for a {num_cols}-column "
            f"board; a variant alphabet must cover every column"
        )
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"column_labels repeats a label: {chosen}")
    if not all(len(ch) == 1 and ch.isalpha() and ch.isupper() for ch in chosen):
        raise ValueError(
            f"column_labels must be single upper-case letters, got {chosen}"
        )
    return chosen


# What the memory block calls a play and what is won or lost, in the rules' own words.
MEMORY_TERMS = MemoryTerms("game", "points")


def make_fns(
    num_rows: int, num_cols: int, spec: Optional[GamePromptSpec] = None
) -> dict:
    """Build the Dark Hex prompt/parse closures for one board shape and spec.

    Action encoding: a = r * num_cols + c (matches OpenSpiel and
    scoring/darkhex_posterior). Column labels A.. left -> right, row
    labels 1.. top -> bottom. Corner verbose names ("top-left" …) exist only at
    2×2, where they are unambiguous, which keeps the canonical 2×2 prompt
    stable.

    :param num_rows: number of board rows, 1 to 9 (int)
    :param num_cols: number of board columns, 1 to 26 (int)
    :param spec: an optional variant ``GamePromptSpec``. The shape comes first because this game
        has one, and the spec is last and always optional. ``None`` reads this module's
        ``_SPEC`` at call time, which keeps the monkeypatch contract the three fixed-size games
        document. A spec passed explicitly is bound here, so a later patch of the module
        attribute does not reach it. The column alphabet is read from the spec's
        ``action_vocabulary`` (see ``_column_letters``), which lets one shape-independent
        variant serve every registered rectangle (Optional[GamePromptSpec])
    :returns: a dict of the ``GamePromptFns`` fields (dict)
    """
    if not (1 <= num_rows <= 9 and 1 <= num_cols <= 26):
        # 9 rows: keeps every parse path unambiguous (the coordinate regex
        # captures single digits - "10,5" silently misparses as (0, 5));
        # 26 cols: chr(ord("A")+c) leaves the alphabet past Z. Registered
        # boards are ≤3; raise loudly rather than emit garbled prompts.
        raise ValueError(
            f"unsupported board {num_rows}x{num_cols}: rows must be 1-9, cols 1-26"
        )

    def _content() -> GamePromptSpec:
        """The spec this call renders from -- see `spec` above. Module-level
        `_SPEC` is read at CALL time when no spec was injected, so the
        documented monkeypatch contract (patch `darkhex._SPEC`) still changes
        what a live match renders."""
        return _SPEC if spec is None else spec

    # Read at factory time, not call time: the label enumeration and the
    # corner-name map are per-instance constants, not per-info-set content.
    vocab = (_SPEC if spec is None else spec).action_vocabulary

    n_cells = num_rows * num_cols
    col_letter = _column_letters(vocab, num_cols)

    action_coords = {
        r * num_cols + c: (r, c) for r in range(num_rows) for c in range(num_cols)
    }
    coords_action = {v: k for k, v in action_coords.items()}
    action_label = {a: f"{col_letter[c]}{r + 1}" for a, (r, c) in action_coords.items()}
    label_action = {v: k for k, v in action_label.items()}

    if (num_rows, num_cols) == (2, 2):
        # Aliasing the module-global
        # `DARKHEX_SPEC.action_vocabulary["corner_names"]` dict directly into
        # this closure would be unsafe -- `frozen=True` on `GamePromptSpec` blocks rebinding
        # `_SPEC.action_vocabulary` to a different object, but does nothing
        # to stop mutating the dict object itself (`corner[a] = ...`), which
        # would leak across every 2x2 `make_fns` call (they'd all share the
        # one dict on the shared `DARKHEX_SPEC` instance) and even into the
        # canonical spec used for `game_spec_hash`/`prompt_spec_hash`. Wrap
        # read-only so any accidental mutation raises instead of leaking.
        corner = types.MappingProxyType(vocab["corner_names"])
        action_longname = {a: f"{action_label[a]} ({corner[a]})" for a in action_coords}
        verbose = {}
        for a, name in corner.items():
            verbose[name] = a
            verbose[name.replace("-", "")] = a
            verbose[name.replace("-", " ")] = a
    else:
        action_longname = {
            a: f"{action_label[a]} (row {r + 1}, column {col_letter[c]})"
            for a, (r, c) in action_coords.items()
        }
        verbose = {}

    def _parse_board(info_state: str) -> list[list[str]]:
        lines = info_state.strip().split("\n")
        board = []
        for r in range(num_rows):
            row_str = lines[r] if r < len(lines) else "." * num_cols
            board.append(list(row_str.ljust(num_cols, ".")))
        return board

    def _parse_move_count_and_history(info_state: str) -> tuple[int, list[int]]:
        """Parse the move-count line and this player's own-move-history line
        that follow the board rows (module docstring lines num_rows /
        num_rows+1). Empirically verified via a live pyspiel walk:
        ``'x.\\n..\\n3\\n0,0 '`` is board ``x.``/``..``, move count 3,
        own history ``[0]`` (a single "player_id,action" pair; player_id is
        always this player's own id in their own info-state -- confirmed by
        walking both P0's and P1's info-states after the same actions).

        Robust to the history line's absence: ``str.strip()`` consumes the
        info-state's trailing newline, so a state where this player has not
        yet acted (e.g. the opening state ``'..\\n..\\n0\\n'``, or the
        opponent's very first info-state) has only ``num_rows + 1`` lines
        after strip+split -- there is no line at index ``num_rows + 1`` to
        read, and history is empty.
        """
        lines = info_state.strip().split("\n")
        move_count = int(lines[num_rows]) if len(lines) > num_rows else 0
        own_actions: list[int] = []
        if len(lines) > num_rows + 1:
            for tok in lines[num_rows + 1].split():
                _player, _sep, action_str = tok.partition(",")
                if action_str:
                    own_actions.append(int(action_str))
        return move_count, own_actions

    def _legal_from_board(board: list[list[str]]) -> list[int]:
        return sorted(
            coords_action[(r, c)]
            for r in range(num_rows)
            for c in range(num_cols)
            if board[r][c] == "."
        )

    def build_prompt(info_state: str, player_id: int) -> str:
        board = _parse_board(info_state)
        legal = _legal_from_board(board)
        move_count, own_actions = _parse_move_count_and_history(info_state)
        own_history_labels = [action_label[a] for a in own_actions if a in action_label]
        history_str = (
            ", ".join(own_history_labels) if own_history_labels else "none yet"
        )
        color = "Black" if player_id == 0 else "White"
        goal = (
            "connect the top row to the bottom row"
            if player_id == 0
            else "connect the left column to the right column"
        )
        own_sym = "x" if player_id == 0 else "o"
        opp_sym = "o" if player_id == 0 else "x"

        def cell(r: int, c: int) -> str:
            ch = board[r][c]
            if ch == own_sym:
                return "[Y]"
            if ch == opp_sym:
                return "[E]"
            return "[ ]"

        header = "     " + "    ".join(col_letter)
        row_lines = [
            f"  {r + 1}  " + "  ".join(cell(r, c) for c in range(num_cols))
            for r in range(num_rows)
        ]
        board_str = "\n".join([header, *row_lines])
        legal_strs = [
            action_longname[a]
            for a in in_display_order(_content(), legal)
            if a in action_longname
        ]

        instr = _content().instruction
        parts = [
            instr["intro"].format(color=color),
            instr["goal"].format(goal=goal),
            "",
            instr["board_header"].format(
                num_rows=num_rows, num_cols=num_cols, last_col=col_letter[-1]
            ),
            instr["legend"],
            board_str,
            "",
            instr["discovery_note"],
            "",
            instr["move_count"].format(move_count=move_count),
            instr["own_history"].format(history_str=history_str),
            "",
            instr["available_moves"].format(
                legal_strs=(", ".join(legal_strs) if legal_strs else "(none)")
            ),
        ]
        return "\n".join(parts) + "\n"

    def action_suffix(info_state: str) -> str:
        board = _parse_board(info_state)
        legal = _legal_from_board(board)
        labels = [
            action_label[a]
            for a in in_display_order(_content(), legal)
            if a in action_label
        ]
        return legal_response_suffix(labels)

    def action_labels(info_state: str) -> dict[int, str]:
        """Legal action -> the exact literal action_suffix offers for it:
        the same `action_label` dict action_suffix
        itself indexes, not a re-derived coordinate scheme. A hand-written
        "{r},{c}" candidate scheme in a different module would drift from
        the labels actually offered, so callers read this dict."""
        board = _parse_board(info_state)
        legal = _legal_from_board(board)
        return labels_in_display_order(
            _content(), {a: action_label[a] for a in legal if a in action_label}
        )

    def parse_action(text: str, info_state: str) -> int | None:
        """Action parser hardening: a completion naming 2+ of the short move
        labels action_labels() offers at this info-set -- e.g. an echo of
        action_suffix, which lists every legal move label -- is rejected
        before any of the matching below (short label, verbose corner name,
        coordinate pair, or bare digit). See `spec.has_multiple_labels`'s
        docstring.
        """
        t = text.strip().lower()
        if has_multiple_labels(text, action_labels(info_state).values()):
            return None
        for label, action in label_action.items():
            if re.search(r"\b" + re.escape(label.lower()) + r"\b", t):
                return action
        for name, action in verbose.items():
            if name in t:
                return action
        m = re.search(r"(\d)\s*[,\s]\s*(\d)", t)
        if m:
            coords = coords_action.get((int(m.group(1)), int(m.group(2))))
            if coords is not None:
                return coords
        # Strict word-bounded single digit - verified 0-diff against the
        # 2x2 \b([0-3])\b acceptance set over a 30k-string corpus:
        # "03"/"12" and letter-adjacent
        # digits ("A3", "card4") stay unparseable -> logged retry/fallback,
        # keeping the n_retries/fallback_used audit trail comparable across
        # runs. A digit-lookaround form is not used: it silently accepts
        # letter-adjacent digits. Bare-int fallback covers actions 0-9 only;
        # labels are the primary path.
        for m in re.finditer(r"\b(\d)\b", t):
            if int(m.group(1)) < n_cells:
                return int(m.group(1))
        return None

    def parse_action_vocab(text: str, info_state: str) -> str | None:
        """Mirrors parse_action exactly to return which vocabulary resolved the text.

        Returns "label" | "verbose" | "coords" | "digit" | None. Branch order matches
        parse_action so the two cannot drift.
        """
        t = text.strip().lower()
        if has_multiple_labels(text, action_labels(info_state).values()):
            return None
        for label, action in label_action.items():
            if re.search(r"\b" + re.escape(label.lower()) + r"\b", t):
                return "label"
        for name, action in verbose.items():
            if name in t:
                return "verbose"
        m = re.search(r"(\d)\s*[,\s]\s*(\d)", t)
        if m:
            if coords_action.get((int(m.group(1)), int(m.group(2)))) is not None:
                return "coords"
        for m in re.finditer(r"\b(\d)\b", t):
            if int(m.group(1)) < n_cells:
                return "digit"
        return None

    def belief_suffix(info_state: str) -> str | None:
        """`belief_exemplar` is a PER-slot
        template (spec.py) with a placeholder key -- it does not list
        `"A1": 0.5, "B2": 0.5, ...` over the real currently-unknown cell
        labels, which would let an echo of belief_suffix's text
        self-parse (every key mapped via `label_action.get`). The `labels`
        line (which cells to answer for) still uses the real legal cells --
        only the JSON-exemplar illustration is placeholder-keyed. This
        closure fills exactly `n_cells` slots (the BOARD's total cell
        count, a per-variant constant fixed once by `make_fns`).

        `n_cells` is not chosen to match the live
        unknown-cell count at this one info-state -- it is a constant for
        the whole variant, chosen specifically to avoid reintroducing true
        per-info-set content variation into the exemplar (tracking the
        live count would do exactly that). Over darkhex_2x3's
        full P0 tree, the 6-key (n_cells) exemplar matches the true live
        unknown-cell count at only 1 of 2,053 info-sets; a fixed-2-key
        version matched more often (1,440 of 2,053), purely
        coincidentally. See `DARKHEX_SPEC.belief_exemplar`'s comment
        (spec.py) for the full measurement. The KEY stays the placeholder
        token `<cell_{i}>` (never a real cell label) at every slot
        regardless, so echo-safety is unaffected by any of this. The
        illustrative values cycle through distinct non-0.5 numbers so the
        exemplar cannot be mistaken for a uniform default."""
        sp = _content()
        board = _parse_board(info_state)
        legal = _legal_from_board(board)
        if not legal:
            return None
        labels = ", ".join(action_label[a] for a in legal)
        # Cycle through distinct non-0.5 values: [0.3, 0.4, 0.6, 0.7, ...]
        illustrative_values = [0.3, 0.4, 0.6, 0.7]
        exemplar = ", ".join(
            sp.belief_exemplar.format(
                i=i, value=illustrative_values[(i - 1) % len(illustrative_values)]
            )
            for i in range(1, n_cells + 1)
        )
        return sp.belief_question.format(
            labels=labels,
            belief_exemplar=exemplar,
            belief_keys=quoted_keys([action_label[a] for a in legal]),
        )

    def parse_belief(text: str, info_state: str) -> str | None:
        board = _parse_board(info_state)
        legal = _legal_from_board(board)
        raw = load_belief_object(text)
        if raw is None:
            return None
        probs: dict[int, float] = {}
        for k, v in raw.items():
            key = str(k).strip().upper()
            a = label_action.get(key)
            if a is None:
                try:
                    a = int(str(k).strip())
                except ValueError:
                    continue
            if a in legal:
                p = finite_float(v)
                if p is None:
                    continue
                probs[a] = min(1.0, max(0.0, p))
        if not probs:
            return None
        return json.dumps({str(a): probs.get(a, 0.0) for a in legal})

    return {
        "build_prompt": build_prompt,
        "action_suffix": action_suffix,
        "action_labels": action_labels,
        "belief_suffix": belief_suffix,
        "parse_action": parse_action,
        "parse_action_vocab": parse_action_vocab,
        "parse_belief": parse_belief,
        "parse_own_card": parse_own_card,
        "memory_terms": MEMORY_TERMS.scaled(
            1 if spec is None else spec.memory_payoff_scale
        ).with_labels(action_labels),
    }


# Canonical 2×2 default instance - module-level names for importers that
# need the default board.
_DEFAULT_FNS = make_fns(2, 2)
build_prompt = _DEFAULT_FNS["build_prompt"]
action_suffix = _DEFAULT_FNS["action_suffix"]
action_labels = _DEFAULT_FNS["action_labels"]
belief_suffix = _DEFAULT_FNS["belief_suffix"]
parse_action = _DEFAULT_FNS["parse_action"]
parse_action_vocab = _DEFAULT_FNS["parse_action_vocab"]
parse_belief = _DEFAULT_FNS["parse_belief"]
