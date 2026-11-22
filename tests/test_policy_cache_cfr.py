"""Cache-once-freeze: CFR policy cache correctness.

Each test names the real failing path it guards:

- ``test_off_path_fidelity_*`` - a cached policy that differed from a fresh solve
  at any info-set (esp. off-path / zero-reach, where the Bayesian likelihood,
  P7b EV rollout, and compute_true_posterior all recurse) would silently corrupt
  the CFRRef agent, the Bayesian discriminator, and the P3/P5/P7 yardstick. This
  is the falsification of the whole approach; it is checked on Leduc (mid-tree
  public chance card) and the wrapped Goofspiel.
- ``test_hit_does_not_resolve`` - if the load path re-solved, the optimisation is
  void (Liar's Dice CFR+ is ~23 min).
- ``test_key_distinct_*`` - a ``short_name`` key would serve one game's / one
  iter-count's policy for another.
- ``test_load_contract_*`` - a version bump or a shape-mismatched blob must force a
  re-solve, never a silent stale/permuted assignment.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pytest

pytest.importorskip("pyspiel")

from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_goofspiel, load_kuhn, load_leduc
from lmopt.games import policy_cache
from lmopt.games.policy_cache import (
    _CFR_ORCHESTRATION_FILE,
    _CFR_SOLVER_FILE,
    _combined_sha256,
    cache_components,
    load_or_solve_cfr,
    paths,
)

#: The same code_hash load_or_solve_cfr computes internally, so a test that
#: reconstructs cache_components() to find/tamper the sidecar it wrote lands
#: on the identical key.
_CODE_HASH = _combined_sha256(_CFR_SOLVER_FILE, _CFR_ORCHESTRATION_FILE)


def _max_abs_diff(game, pol_a, pol_b) -> float:
    """Max |Δ action_prob| between two policies over the entire game tree.

    Recurses every legal action and every chance outcome, so it visits off-path /
    zero-reach info-sets - not just the equilibrium-reach nodes.
    """
    md = 0.0

    def walk(s):
        nonlocal md
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        p = s.current_player()
        a1 = pol_a.action_probabilities(s, p)
        a2 = pol_b.action_probabilities(s, p)
        for k in set(a1) | set(a2):
            md = max(md, abs(a1.get(k, 0.0) - a2.get(k, 0.0)))
        for a in s.legal_actions():
            walk(s.child(a))

    walk(game.new_initial_state())
    return md


@pytest.mark.parametrize(
    "loader", [load_leduc, load_goofspiel], ids=["leduc", "goofspiel"]
)
def test_off_path_fidelity(loader, tmp_path):
    game = loader()
    iters = 50
    # Miss (solves + persists), then hit (loads).
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    cached = load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    # A fresh, independent solve (CFR+ is deterministic) is the ground truth.
    fresh = solve_cfr(game, iters, plus=True)
    assert _max_abs_diff(game, cached, fresh) == 0.0


def test_hit_does_not_resolve(tmp_path, monkeypatch):
    game = load_kuhn()
    iters = 40
    calls = {"n": 0}
    real = solve_cfr

    def counting(g, it, plus=True):
        calls["n"] += 1
        return real(g, it, plus=plus)

    # load_or_solve_cfr does a lazy `from lmopt.games.cfr import solve_cfr`, so
    # patching the attribute on that module is picked up at call time.
    monkeypatch.setattr("lmopt.games.cfr.solve_cfr", counting)

    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # miss
    assert calls["n"] == 1
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # hit
    assert calls["n"] == 1  # Not re-solved


def test_key_distinct_by_iters(tmp_path):
    game = load_kuhn()
    load_or_solve_cfr(game, 30, plus=True, cache_dir=tmp_path)
    load_or_solve_cfr(game, 40, plus=True, cache_dir=tmp_path)
    blobs = sorted(tmp_path.glob("cfr_*.pkl"))
    assert len(blobs) == 2  # distinct iteration counts => distinct files


def test_key_distinct_by_game_params(tmp_path):
    # Goofspiel num_cards is embedded in str(game); it must key distinctly so a
    # larger variant never reuses the small-game policy.
    g4 = load_goofspiel(num_cards=4)
    g5 = load_goofspiel(num_cards=5)
    c4 = cache_components(g4, iters=20, solver="cfr+", code_hash=_CODE_HASH)
    c5 = cache_components(g5, iters=20, solver="cfr+", code_hash=_CODE_HASH)
    assert c4["game"] != c5["game"]
    assert paths(tmp_path, "cfr", c4)[0] != paths(tmp_path, "cfr", c5)[0]


def test_load_contract_version_bump_forces_resolve(tmp_path, monkeypatch):
    game = load_kuhn()
    iters = 30
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # writes cache
    comps = cache_components(game, iters=iters, solver="cfr+", code_hash=_CODE_HASH)
    _, meta_path = paths(tmp_path, "cfr", comps)
    # Tamper the sidecar version: current comps no longer match => must miss.
    with open(meta_path) as f:
        meta = json.load(f)
    meta["pyspiel_version"] = "0.0.0-stale"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    calls = {"n": 0}
    real = solve_cfr

    def counting(g, it, plus=True):
        calls["n"] += 1
        return real(g, it, plus=plus)

    monkeypatch.setattr("lmopt.games.cfr.solve_cfr", counting)
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    assert calls["n"] == 1  # stale version => re-solved, not served


def test_load_contract_bad_shape_never_assigned(tmp_path):
    game = load_leduc()
    iters = 30
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # writes cache
    comps = cache_components(game, iters=iters, solver="cfr+", code_hash=_CODE_HASH)
    blob_path, _ = paths(tmp_path, "cfr", comps)
    # Corrupt the blob with a wrong-shape array but keep the sidecar valid.
    with open(blob_path, "wb") as f:
        pickle.dump(np.zeros((3, 2)), f)
    # Must fall through to a fresh solve (shape guard), never assign the bad array.
    recovered = load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    fresh = solve_cfr(game, iters, plus=True)
    assert _max_abs_diff(game, recovered, fresh) == 0.0


def test_load_contract_nonarray_blob_resolves(tmp_path):
    # A valid pickle of a NON-array (e.g. a dict) with a matching sidecar must
    # re-solve, not raise AttributeError on `array.shape`.
    game = load_kuhn()
    iters = 30
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # writes cache
    comps = cache_components(game, iters=iters, solver="cfr+", code_hash=_CODE_HASH)
    blob_path, _ = paths(tmp_path, "cfr", comps)
    with open(blob_path, "wb") as f:
        pickle.dump({"not": "an array"}, f)
    recovered = load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    fresh = solve_cfr(game, iters, plus=True)
    assert _max_abs_diff(game, recovered, fresh) == 0.0


def test_code_change_forces_a_resolve_not_a_stale_serve(tmp_path, monkeypatch):
    """A cache written under one version of cfr.py must read
    as a miss, not a corrupt-blob warning or a stale HIT, once cfr.py's bytes
    change -- the code-identity component of the cache key, not a shape or
    version check, is what catches this.
    """
    solver_file = tmp_path / "cfr_stand_in.py"
    solver_file.write_text("# version 1\n")
    monkeypatch.setattr(policy_cache, "_CFR_SOLVER_FILE", solver_file)

    game = load_kuhn()
    iters = 25
    calls = {"n": 0}
    real = solve_cfr

    def counting(g, it, plus=True):
        calls["n"] += 1
        return real(g, it, plus=plus)

    monkeypatch.setattr("lmopt.games.cfr.solve_cfr", counting)

    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # miss, solves
    assert calls["n"] == 1
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # hit, no solve
    assert calls["n"] == 1

    solver_file.write_text("# version 2, unrelated to the CFR loop itself\n")
    load_or_solve_cfr(
        game, iters, plus=True, cache_dir=tmp_path
    )  # code changed => miss
    assert (
        calls["n"] == 2
    ), "a code change must force a re-solve, not serve the old blob"


def test_orchestration_file_change_forces_a_resolve_too(tmp_path, monkeypatch):
    """CFR's cache ORCHESTRATION (whether a blob is a hit, how iterations/plus
    reach solve_cfr) lives in policy_cache.py itself, not cfr.py -- so the CFR
    code_hash must change when that file changes too, not only cfr.py.
    """
    orchestration_file = tmp_path / "policy_cache_stand_in.py"
    orchestration_file.write_text("# version 1\n")
    monkeypatch.setattr(policy_cache, "_CFR_ORCHESTRATION_FILE", orchestration_file)

    game = load_kuhn()
    iters = 25
    calls = {"n": 0}
    real = solve_cfr

    def counting(g, it, plus=True):
        calls["n"] += 1
        return real(g, it, plus=plus)

    monkeypatch.setattr("lmopt.games.cfr.solve_cfr", counting)

    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # miss, solves
    assert calls["n"] == 1
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)  # hit, no solve
    assert calls["n"] == 1

    orchestration_file.write_text(
        "# version 2, simulating a load_or_solve_cfr change\n"
    )
    load_or_solve_cfr(
        game, iters, plus=True, cache_dir=tmp_path
    )  # code changed => miss
    assert (
        calls["n"] == 2
    ), "a change to policy_cache.py's own orchestration must force a re-solve too"


def test_unchanged_code_still_hits_across_process_style_reloads(tmp_path, monkeypatch):
    """Positive anchor to pair with test_code_change_forces_a_resolve_not_a_stale_serve:
    that test proves a change is caught; this proves an unchanged file keeps
    hitting, so the code-identity component is not accidentally unstable
    (e.g. keyed on a file mtime or a fresh object id) across two independent
    computations of the same hash.
    """
    solver_file = tmp_path / "cfr_stand_in.py"
    solver_file.write_text("# version 1\n")
    monkeypatch.setattr(policy_cache, "_CFR_SOLVER_FILE", solver_file)

    game = load_kuhn()
    iters = 25
    calls = {"n": 0}
    real = solve_cfr

    def counting(g, it, plus=True):
        calls["n"] += 1
        return real(g, it, plus=plus)

    monkeypatch.setattr("lmopt.games.cfr.solve_cfr", counting)

    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    load_or_solve_cfr(game, iters, plus=True, cache_dir=tmp_path)
    assert calls["n"] == 1


def test_load_or_solve_cfr_never_opens_blob_path_directly_for_writing(
    tmp_path, monkeypatch
):
    """load_or_solve_cfr's cache-miss write path must publish the CFR blob
    via a temp file + os.replace, never by opening blob_path itself in a
    write mode -- a reader could otherwise observe blob_path mid-write
    (truncated or partially written), exactly the corruption mechanism
    the atomic publish closes.
    """
    game = load_kuhn()
    cache_dir = str(tmp_path)
    comps = cache_components(game, iters=2, solver="cfr+", code_hash=_CODE_HASH)
    blob_path, _ = paths(cache_dir, "cfr", comps)
    real_open = open

    def guarded_open(file, mode="r", *a, **kw):
        if str(file) == blob_path and any(c in mode for c in "wxa"):
            raise AssertionError(
                f"load_or_solve_cfr opened {blob_path!r} directly in write "
                f"mode {mode!r} -- it must only ever write a temp file and "
                "publish via os.replace, so blob_path itself is never "
                "observably truncated/partial"
            )
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr("builtins.open", guarded_open)
    result = load_or_solve_cfr(game, 2, plus=True, cache_dir=cache_dir)

    fresh = solve_cfr(game, 2, plus=True)
    assert _max_abs_diff(game, result, fresh) == 0.0


def test_load_or_solve_cfr_publishes_the_blob_via_os_replace(tmp_path, monkeypatch):
    """Positive anchor to pair with
    test_load_or_solve_cfr_never_opens_blob_path_directly_for_writing: that
    test only proves blob_path is never opened directly (a negative
    assertion bypassable by any writer that reaches the filesystem some
    other way -- os.open, Path.write_bytes, a C-level writer -- none of
    which touch builtins.open). This test proves the positive half: the
    only way blob_path's content ever changes is via exactly one
    os.replace(..., blob_path) call.
    """
    game = load_kuhn()
    cache_dir = str(tmp_path)
    comps = cache_components(game, iters=2, solver="cfr+", code_hash=_CODE_HASH)
    blob_path, _ = paths(cache_dir, "cfr", comps)
    real_replace = os.replace
    replace_calls = []

    def recording_replace(src, dst, *a, **kw):
        replace_calls.append((src, dst))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", recording_replace)
    load_or_solve_cfr(game, 2, plus=True, cache_dir=cache_dir)

    blob_replacements = [c for c in replace_calls if str(c[1]) == blob_path]
    assert len(blob_replacements) == 1, (
        f"expected exactly one os.replace(..., {blob_path!r}) call publishing "
        f"the blob, got {blob_replacements!r} (all calls: {replace_calls!r})"
    )
    src, dst = blob_replacements[0]
    assert dst == blob_path
    assert src != blob_path, "the replace source must be a distinct temp path"
    assert os.path.exists(blob_path), "the published blob must exist after the call"


def test_write_meta_publishes_via_os_replace(tmp_path, monkeypatch):
    """Positive anchor to pair with
    test_write_meta_never_opens_meta_path_directly_for_writing, same
    reasoning as test_load_or_solve_cfr_publishes_the_blob_via_os_replace.
    """
    meta_path = str(tmp_path / "meta.json")
    real_replace = os.replace
    replace_calls = []

    def recording_replace(src, dst, *a, **kw):
        replace_calls.append((src, dst))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", recording_replace)
    policy_cache._write_meta(meta_path, {"a": 1})

    meta_replacements = [c for c in replace_calls if str(c[1]) == meta_path]
    assert len(meta_replacements) == 1, (
        f"expected exactly one os.replace(..., {meta_path!r}) call publishing "
        f"the sidecar, got {meta_replacements!r} (all calls: {replace_calls!r})"
    )
    src, dst = meta_replacements[0]
    assert dst == meta_path
    assert src != meta_path, "the replace source must be a distinct temp path"
    with open(meta_path) as f:
        assert json.load(f) == {"a": 1}


def test_write_meta_never_opens_meta_path_directly_for_writing(tmp_path, monkeypatch):
    """Same property as
    test_load_or_solve_cfr_never_opens_blob_path_directly_for_writing, for
    the JSON sidecar written by _write_meta.
    """
    meta_path = str(tmp_path / "meta.json")
    real_open = open

    def guarded_open(file, mode="r", *a, **kw):
        if str(file) == meta_path and any(c in mode for c in "wxa"):
            raise AssertionError(
                f"_write_meta opened {meta_path!r} directly in write mode "
                f"{mode!r} -- it must only ever write a temp file and "
                "publish via os.replace, so meta_path itself is never "
                "observably truncated/partial"
            )
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr("builtins.open", guarded_open)
    policy_cache._write_meta(meta_path, {"a": 1})

    with open(meta_path) as f:
        assert json.load(f) == {"a": 1}


def test_hole_in_raw_buffer_tail_passes_shape_and_type_check_silently():
    """A byte-level hole confined to numpy's raw-buffer tail (never
    touching the pickle header/shape/dtype metadata) round-trips as a
    correctly-typed, correctly-shaped array with silently wrong values.
    This is why load_or_solve_cfr's isinstance+shape check on read cannot,
    by itself, catch every corruption an unsynchronized concurrent write
    can produce -- the write itself must be atomic, not merely checked on
    load.
    """
    rng = np.random.default_rng(0)
    arr = rng.random((500, 4))
    arr /= arr.sum(axis=1, keepdims=True)

    full = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
    tail_start = len(full) - arr.nbytes  # numpy's header/shape/dtype
    # metadata occupies everything before this offset; the raw contiguous
    # float buffer occupies [tail_start:].
    hole_start = tail_start + (len(full) - tail_start) // 3
    hole_end = tail_start + 2 * (len(full) - tail_start) // 3

    corrupted = bytearray(full)
    for i in range(hole_start, hole_end):
        corrupted[i] = 0
    corrupted = bytes(corrupted)

    result = pickle.loads(corrupted)  # must not raise

    assert isinstance(result, np.ndarray)
    assert result.shape == arr.shape
    assert result.dtype == arr.dtype
    assert not np.array_equal(result, arr), (
        "expected the hole to have silently corrupted values -- if this "
        "now fails, numpy's pickle layout changed and the mechanism this "
        "test documents needs re-deriving, not just re-asserting"
    )
