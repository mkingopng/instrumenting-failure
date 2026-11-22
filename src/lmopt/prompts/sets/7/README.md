# Prompt set 7

Every row collected at `prompt_schema_version = 7` was rendered from this text. The set is
frozen: a change to any string here is a new set.

## What is in it

| File | Holds |
| --- | --- |
| `kuhn.json`, `leduc.json`, `goofspiel.json`, `liarsdice.json`, `darkhex.json` | one game's content: `instruction` (the rules paragraph and the situational lines the game module fills per state), `action_vocabulary` (the reply labels and parse-only synonyms), `belief_question`, `belief_exemplar`, `display_scale` |
| `wrappers/instruct.json` | the sentences every game shares: the objective paragraph, the three-step belief update, the opponent assumption, the belief format sentences, and the legal-response sentence that closes every action call |
| `memory.json` | the cross-hand memory wording (M1 summary, M2 transcript line, M3 notes header, the amount format), as `str.format` templates |
| `variants/relabel.json` | what the relabel run changes per game: new label literals and synonyms (Kuhn, Leduc, Liar's Dice), the rotated column alphabet and the two prose spans it touches (Dark Hex), the number-word card labels and the two prose spans (Goofspiel) |
| `variants/payoff_scale.json` | the x10 factor; per game, the exact prose spans scaled with the count each must appear, and whether the game also scales magnitudes rendered from state |
| `rendered/<game>.<kind>.txt` | the prompt a model sees at each game's first decision state: `action_m0`, `belief_m0`, and `action_m1`/`action_m2` with a fixed four-hand memory. |

A content string is a literal or a list of parts; a part is a literal or
`{"wrapper": "<key>", "prefix": "...", "fields": {...}}`, composed by concatenation
(`lmopt.prompts.compose`). The content hash over the five composed specs is
`3dcdecc6b754d0e736d934ae5e9944d04477967e75289a188b35f637d5151f9a` (`PROMPT_SPEC_HASHES[7]`),
and the memory wording hash is `b5b3d4ec09b8d4d2e0ddbaf5b23e4ce4571914917cfd32842078cf518c0f67d0`;
both are pinned by tests. `corner_names` and `card_labels` are keyed by action index, which
JSON stores as strings; the loader reads them back as integers.

## What it states

Every game's rules state what the engine does and every magnitude a model can be scored on:
Liar's Dice says a six is wild and names the highest bid; Kuhn states the ante and the 1 and 2
chip payoffs; Leduc the ante and the 2 and 4 chip bet sizes; Goofspiel that the result is the
point difference; Dark Hex the 1 point stake. Every game states the zero-sum objective. Every
action call closes with one shared sentence naming the exact legal replies at that node. Every
belief call asks for the posterior in three numbered steps, names its keys in a plain sentence
(not a parseable object, so an echo cannot parse as a belief), keeps placeholder keys in the
exemplar, and asks for 2 decimal places summing to about 1. The memory block names what it is in
each game's own words ("Your match so far, against the same opponent in every hand:", hands or
games, chips or points, Goofspiel's payoff in the units its rules state); `memory_render_hash()`
records which memory wording a run collected under.

## What collected under it

Experiment 7: Qwen, OLMo and Mistral (the ~14B trio) at temperature 0.0, memory levels M0,
M1 and M2, prompt version 7 in every row. Experiment 8 is this same set at temperature 0.2.
