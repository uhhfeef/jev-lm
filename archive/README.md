# What was tried and cut

`jev_letter_lm.py` used to carry four stopping mechanisms, three sampling knobs
and three identity modes, most of them behind flags that no call site ever set
away from their defaults. Each one was measured before it was removed. This
directory keeps the code and the measurements so none of it has to be rediscovered.

- **`removed_mechanisms.py`** — the removed code itself, sectioned, with the
  number that killed each piece in the comment above it.
- **`probe_stop.py`, `probe_results.json`** — the stop-mechanism probe: eight
  hand-labelled prefixes, four mechanisms, scored on how well each separates a
  complete answer from an unfinished one.
- **`ab_test.py`, `ab_results.json`** — the end-to-end A/B over the same four
  arms. Only the `gate` arm finished before the probe made the rest redundant.

Both scripts still import names that `jev_letter_lm.py` no longer exports, so
they do not run as-is; the definitions they want are all in
`removed_mechanisms.py`.

## Stopping: gate vs parallel vs symbol

How the answer decides it is finished. `gate` asks a separate `should_stop`
question first and only pays for the character if it does not fire; `parallel`
batches that question alongside the character; `symbol` has no gate at all —
STOP is a 28th option in the Choice, its hypothesis being the answer with nothing
appended, so "this is finished" is drawn from the same distribution as every letter.

| mechanism | unfinished prefixes | complete prefixes | separation | requests/char |
|---|---|---|---|---|
| gate | 0.01–0.14 | 0.70–0.89 | +0.76 | 2 |
| parallel | 0.01–0.16 | 0.65–0.90 | +0.75 | 1 |
| **symbol** | **0.00–0.03** | **0.92–0.99** | **+0.95** | **1** |

`symbol` wins on both axes, so `gate` and `parallel` are gone along with the
`SHOULD_STOP` noul, `stop_state()`, the threshold, and the second copy of the
character instructions.

The extra request bought nothing: a gate that cannot see how the alphabet is
leaning is judging the same text with strictly less information than the
distribution it is supposed to be gating.

## Shaping STOP: `--stop-bias`, `--min-steps`

The worry was that hypothesis options inflate STOP, since the unchanged answer is
a valid prefix of every extension of itself. `--stop-bias 0.5 --min-steps 3` was
the correction, scored as the `symbol+` arm:

| `answer_so_far` | complete? | symbol | symbol+ |
|---|---|---|---|
| `""` | no | 0.010 | 0.005 |
| `"par"` | no | 0.000 | 0.000 |
| `"paris is"` | no | 0.027 | 0.014 |
| `"paris"` | yes | 0.950 | 0.905 |
| `"the capital of france is paris"` | yes | 0.977 | 0.954 |
| `"je"` | no | 0.000 | 0.000 |
| `"jev"` | yes | 0.917 | 0.846 |
| `"my name is jev"` | yes | 0.987 | 0.974 |

The correction is real but points the wrong way. The unfinished side was already
at 0.00–0.03, so there was no inflation left to discount, and every complete
prefix lost 4–7 points for it. The empty answer scores 0.01 unshaped, so
`--min-steps` guards against something that does not happen.

## Sampling: `--temperature`, `--top-k`, `--top-p`

A question with one correct answer gains nothing from variety, and a
character-level answer cannot recover from a wrong character, so every degree
above 0 is a fresh chance to lose with no upside. Asked for the largest planet,
Jev ranks `t=0.31` over `e=0.22` at `"jupi"` — the right letter — and sampling at
0.4 still draws `e` 28% of the time, which is `jupiesne.` instead of `jupiter.`

`--temperature` therefore defaulted to 0 and every call site left it there, which
left `--top-k` and `--top-p` sitting below a branch that never executed. The loop
now calls `argmax` directly.

## Identity: `--identity full` / `unspelled`

Sending "spelled j, e, v" on every request taught the model to write it: asked for
the largest planet it answered `jev pa .`, taking `e=0.35` over `u=0.15` at the
second character of *jupiter*. `unspelled` dropped the spelling everywhere, which
costs the one question it exists for; `gated` sends it only when the question is
about the model, and is now the only behaviour.

## What survived, and why

- **Beam search.** `beam2_results.json`: beam width 3 at length penalty 0.7 is
  correct on 3/3 seeds against greedy's 2/3, at ~3× the requests. Greedy's loss is
  the mechanism the beam exists for — it commits to a locally-attractive character
  and cannot revisit it.
- **`--ensemble`.** Several shuffled option orderings averaged per step, batched
  into one request, so Jev's position bias cancels instead of deciding the answer.
  Free in requests. `probe_ensemble.py` measures it.
- **`--window`, `--seed`, `--max-chars`, `--quiet`.**
