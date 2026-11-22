# Instrumenting Failure

Code for the NeurIPS 2026 workshop paper *Instrumenting Failure: Scoring Language Model Agents at Every
Decision Node in Imperfect-Information Games* (Interpreting Agent Behavior workshop).

The harness plays open-weight language models against a ladder of game-theoretic agents in five zero-sum,
imperfect-information games, logs every decision node, and scores the play offline against a CFR+
equilibrium reference along eight optimality properties. [OpenSpiel](https://github.com/google-deepmind/open_spiel)
is the single source for both the game environment and the equilibrium reference. This repository holds
what is needed to reproduce the paper's two collections (experiment 7 at temperature 0 and experiment 8 at
temperature 0.2, both under prompt set 7) and the analysis behind every number in the paper.

## Layout

| Path | What it holds |
|---|---|
| `src/lmopt/` | The package: games and CFR references, agents, prompting, elicitation, match harness, logging, scorers and analysis |
| `src/lmopt/prompts/sets/7/` | The prompt text every row in the paper was collected under, with rendered examples |
| `scripts/` | Entry points: reference solves, collection, acceptance gates, scoring, scoreboard |
| `configs/` | Job templates for experiments 7 and 8, the model roster, anchors and the board manifests that pin the scored outputs |
| `tests/` | A core test set, including the semi-synthetic coverage checks |

## Games, opponents and models

| Game | OpenSpiel game string |
|---|---|
| Iterated Kuhn | `kuhn_poker` (iterated at match level) |
| Leduc | `leduc_poker` |
| Goofspiel | `turn_based_simultaneous_game(game=goofspiel(imp_info=true,num_cards=4,points_order=descending,returns_type=point_difference))` |
| Liar's Dice | `liars_dice(numdice=1,dice_sides=6)` |
| Dark Hex | `dark_hex(board_size=2)` |

The canonical strings live in `src/lmopt/games/loaders.py`. Opponents are the CFR+ reference, an exact
Bayesian best responder, a self-play RL agent and an always-bet baseline. The three models (Qwen2.5-14B-Instruct,
OLMo-2-1124-13B-Instruct, Mistral-Nemo-Instruct-2407) and their pinned revisions are in `configs/models.yaml`.

The eight properties are P1 invariance, P2 adaptivity, P3 randomisation, P4 payoff sensitivity, P5 calibration,
P6 sequential rationality, P7 Bayesian consistency (P7a belief error, P7b belief-action consistency) and
P8 value of information. The scorers are in `src/lmopt/scoring/`.

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/) are assumed.

```bash
make install        # uv sync --extra dev --extra tracking
make install-serve  # the same plus vLLM (Linux only)
```

The harness, scoring and analysis install from wheels on Linux (x86_64, aarch64), macOS (Apple silicon) and
Windows (x86_64). OpenSpiel (`open_spiel`) has no wheel for Intel macOS or Windows on ARM and builds from
source there, which needs CMake and a C++ toolchain. vLLM is in the `serve` extra and installs on Linux only
(glibc 2.34 or newer for a wheel-only install). Tests, scoring and analysis run on any of the platforms above;
collection with the shipped job templates needs Linux, because the harness and the vLLM server run on the same
machine and the harness inspects the server process to record its versions. Collection uses POSIX file locks (`fcntl`), so on Windows run it
under WSL. The Makefile targets assume a POSIX shell and `.venv/bin/python`.

Run scripts with the project interpreter and the repository on the path, as the Makefile does:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/<script>.py --help
```

## Gates

```bash
make test               # the core test set
make validate-cfr       # solves CFR+ per game; exits non-zero on a tolerance breach or if Kuhn departs from -1/18
make validate-coverage  # semi-synthetic coverage and mutation checks for the P5, P7a and P8 intervals (slow)
make validate-lp        # cross-checks the references against a linear-program solution
```

`make validate-cfr` takes hours unfiltered; `scripts/validate_cfr.py --game <key>` runs one game.
Collection has its own gates: `scripts/pilot_gate.py` (a pilot must pass before full jobs run), and the
exact-policy pass (`scripts/exact_policy_pass.py`).

## Reproducing the experiments

Each step's script documents its arguments under `--help`.

**Serving.** `make install-serve` installs vLLM 0.31.0 (with torch 2.13.0) on the GPU machine, and the job
templates declare that stack under `serving.serving_env` (`vllm: "0.31.0"`, `torch: "2.13.0"`, `python: "3.13"`).
Before collecting, the harness reads the running server's own versions and refuses if they differ from the
declaration, so a run is never stamped with the wrong stack. vLLM and torch must match exactly; Python is declared
by minor version, so any 3.13 patch release passes. Keep one patch release for all the seeds of a condition,
though: the observed full version is recorded per run, and composing seeds refuses runs whose recorded serving
environments differ. The versions the server actually reports, including the full
Python version, are what each run's metadata records. To serve with a different stack, edit the values in the
templates under `configs/`. The paper's own data was served with an earlier stack (vLLM 0.18.0, torch 2.10.0,
Python 3.13.2).

**1. Warm the reference caches.** Every CFR and RL reference is solved once and cached; parallel runs refuse
a cold cache.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/precompute_references.py --cfr-iters 1000 --rl-iters 1000
```

The default is the paper's five games. `--games` also accepts larger variants that are not part of the paper
and take far longer to solve.

**2. Collect.** One job is one seed of one model on one GPU. `scripts/collection_job.py` holds the model, job-kind
and size tables and renders each job's config from the templates in `configs/`. `render` writes the configs and
prints shell assignments: the model id and revision to serve, one sweep command per config, and the pilot-gate
command where the job has one.

```bash
# the rendered commands call a bare `python`, so activate the environment and put the checkout on the path
source .venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD
# what a job would run, with its size and time estimate
python scripts/collection_job.py plan --model qwen --job v8_m0 --index 0
# write the job's configs and load its commands
eval "$(python scripts/collection_job.py render --model qwen --job v8_m0 --index 0 --port 8000 --out /tmp/job-configs)"
# serve the model at the pinned revision, wait until it answers, then run each sweep (and the gate, when GATE=1)
vllm serve "$MODEL_ID" --revision "$REVISION" --port 8000 --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" &
eval "$SWEEP_CMD_0"
[ "$GATE" = 1 ] && eval "$GATE_CMD"
```

Run the pilot of a memory level first; full jobs are meant to follow a passing pilot. A pilot takes no `--index`:
`python scripts/collection_job.py render --model qwen --job v8_pilot_m0 --port 8000 --out /tmp/job-configs`.

Experiment 8 job kinds carry a `v8_` prefix: `v8_m0`, `v8_m1`, `v8_m2` (memory levels, seed index 0-9),
`v8_relabel`, `v8_payoff_scale`, `v8_offpath` and their `_m1`/`_m2` forms (the runs P1, P4 and P6 need, index 0-4),
`v8_m1_null`, `v8_m2_null` and the pilots `v8_pilot_m0|m1|m2`. Experiment 7 uses the same names without the prefix;
`plan` refuses those unless `LMOPT_ALLOW_EXPERIMENT_7=1` is set in its environment (a guard against typing `m0`
for `v8_m0`), while `render` does not check. Outputs land under
`data/run_experiment/recycle-v<7|8>-*/`.

Two things to get right before collecting:

- **A committed checkout.** Every row records the commit of the code that produced it, and each run records any
  uncommitted paths under `src/`, `scripts/` and `configs/`. Collect from a git checkout of this repository
  with those directories committed; an unpacked archive with no `.git` records no commit.
- **The serving environment.** See Serving above: the job templates declare the server's versions and the
  harness refuses a server that differs.

**Accept.** A finished run is checked against the kill criteria and stamped as accepted; composing and scoring
read only accepted runs. `scripts/ladder_state.py` reports which runs exist and which are accepted.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/recycle_arm_evidence.py <run dir> --config <the job's rendered config> --stamp
PYTHONPATH=src:. .venv/bin/python scripts/ladder_state.py
```

**Anchors.** `configs/anchors/` holds the reference points the scorers read: the uniform-random agent's P3 regret
and P6 off-path distance, and the adaptive agent's P2 ceiling. They are rebuilt without a model or GPU by running
`scripts/run_sweep.py --config` on `configs/p3_sentinel.yaml`, `configs/p6_sentinel/<game>.yaml` or
`configs/p2_adaptive_reference.yaml`, then the matching `scripts/build_p*_anchors.py`.

**3. Compose and score.** Seeds are composed into one condition per model and memory level, then scored.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/compose_arm_pieces.py --piece <seed dir> --piece <seed dir> ... --out <composite dir>
PYTHONPATH=src:. .venv/bin/python scripts/analyse_stats.py \
  --data data/run_experiment/recycle-v<N>-<level>/composite-<model> \
  --out analysis/recycle-v<N>-all/<model>-<level> \
  --cfr-iters 1000 --bootstrap 2000 --require-cache-hit
PYTHONPATH=src:. .venv/bin/python scripts/probe_scores.py --base <arm dir> --relabel <relabel arm dir> \
  --payoff <payoff-scale arm dir> --driven <off-path arm dir> --out analysis/recycle-v<N>-all/<model>-<level>
PYTHONPATH=src:. .venv/bin/python scripts/backfill_condition.py --analysis-root analysis/recycle-v<N>-all
PYTHONPATH=src:. .venv/bin/python scripts/arm_collapse_pass.py --analysis-root analysis/recycle-v<N>-all \
  --offpath-root data/run_experiment/recycle-v<N>-probes/composite-offpath/seeds<A>-<B>
```

**4. Scoreboard.** `scripts/scoreboard.py` renders the three 120-cell boards (3 models x 5 games x 8 properties,
one board per memory level) from a scored root and writes them between the `<!-- SCOREBOARD:BEGIN -->` and
`<!-- SCOREBOARD:END -->` markers of a markdown file. `--expect-arms` is required: it is the number of scored
conditions the root should hold (9 for three models at three memory levels).

```bash
printf '<!-- SCOREBOARD:BEGIN -->\n<!-- SCOREBOARD:END -->\n' > scoreboard.md
PYTHONPATH=src:. .venv/bin/python scripts/scoreboard.py --analysis-root analysis/recycle-v<N>-all \
  --expect-arms 9 --board-file scoreboard.md
```

A root that holds only some memory levels needs `--partial`. `--coverage-floor configs/anchors/paired_coverage_floor.json`
applies the pre-registered coverage floor to the P1 and P4 cells.

`configs/board_manifests/` holds the sha256 manifests of the two scored roots the paper reads
(`recycle-v7-all-20261021`, `recycle-v8-all-20261012`). They pin the published scored data: check a downloaded
copy with

```bash
PYTHONPATH=src:. .venv/bin/python scripts/board_manifest.py --root analysis/<root> --verify configs/board_manifests/<root>.json
```

A root re-scored with this code reproduces the numbers but not every byte: some note and label text written
into the scored files was reworded for this release, so those files hash differently from the manifests.

## Data

Collected rows (one Parquet row per decision node) and the scored analysis roots are not in this repository.

## Licence

Apache-2.0; see `LICENSE`.
