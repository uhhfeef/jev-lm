# jev-lm

A character-level language model built out of a classifier.

Jev is a System One model from TypeSafe: it picks one option from a defined set
and returns a calibrated probability over that set. That is the same output shape
as a character-level language model's softmax. So the vocabulary becomes 28
Choice options — `a-z`, space, and stopping — and generation is the ordinary
autoregressive loop: ask, take the most probable option, append, ask again.

```
$ jev-lm
ask a question. blank line to skip, 'exit' or ctrl-c to leave.

you> what is the capital of france
jev> paris.
```

The answer arrives one character at a time, because that is how it is written.
`--verbose` shows the distribution behind each one:

```
$ jev-lm --question "what is the capital of france" --seed 1 --verbose
"p"        | stop=0.01 | p=0.27 a=0.22 w=0.20 t=0.12 c=0.11
"pa"       | stop=0.01 | a=0.60 o=0.14 space=0.06 r=0.05 i=0.04
"par"      | stop=0.00 | r=0.88 s=0.04 i=0.03 space=0.01 l=0.01
"pari"     | stop=0.00 | i=0.80 s=0.09 a=0.03 space=0.03 e=0.02
"paris"    | stop=0.00 | s=0.99 space=0.01 h=0.00 p=0.00 w=0.00
"paris."   | stop=0.97 | STOP=0.97 space=0.03 s=0.00 o=0.00 b=0.00
jev> paris.
```

The first character is the hardest one: `p` leads a-w-t-c by six points and the
rest of the word follows at 0.80 or better once it is down.

## Demos

Two short clips of the session, recorded against the current code:

- [`docs/session-whats-up.mp4`](docs/session-whats-up.mp4) — `whats up` -> `what up .`
- [`docs/session-who-are-you.mp4`](docs/session-who-are-you.mp4) — `who are you?` -> `is a man .`

Both show the loop doing what it is built to do and answering badly, which is the
honest state of it on open-ended questions: the classifier is picking the most
plausible next character, and with nothing factual to anchor on it drifts into a
near-echo of the question. The closed questions in the traces above are where it
holds up.

## What makes it work

**Options are hypotheses, not symbols.** Each option is the text you would have
if you picked it — `"my name is jev"` — never a description of a symbol like
"the next letter is 'v'". Candidate answers differ from each other in meaning,
which is what a classifier can judge; near-identical sentences about letters do
not. This is the single largest thing separating a working loop from a broken one.

**Every request carries the question.** Without a target, a classifier has
nothing to discriminate on: "which character reads most natural" has no answer,
while "which character best begins a correct answer to `question`" does.

**Stopping is the 28th option.** Its hypothesis is the answer with nothing added
to it, so "this is finished" is drawn from the same distribution as every letter.
Asking a separate `should_stop` question first was tried and lost on both axes —
weaker separation, twice the requests.

**Each step averages several shuffled orderings.** Jev leans toward whichever
options come first in the criteria map, and one shuffle per step only randomises
that lean. Several orderings batched into one request cancel it, at no extra
round trips.

## Install

```bash
uv venv
uv pip install -e ".[dev,experiments]"
cp .env.example .env    # then fill in TYPESAFE_API_KEY
```

## Use

```bash
jev-lm                                              # a session: ask, answer, repeat
jev-lm --question "what is the capital of france"   # answer one and exit
jev-lm --question "largest planet" --beam-width 3
```

A session holds one client and one `Random` for its whole life, and each question
is independent: `question` is fixed for a run by construction and the state carries
only `answer_so_far`, so there is nothing to thread between turns. `exit`, `quit`,
ctrl-c or ctrl-d leaves; a failed request ends that answer, not the session.

| flag | default | |
|---|---|---|
| `--question` | none | answer one question and exit; omit for a session |
| `--max-chars` | 60 | character budget |
| `--window` | 40 | how much of the answer each option shows |
| `--ensemble` | 3 | shuffled orderings averaged per step; free in requests |
| `--beam-width` | 1 | candidates kept alive; >1 costs one request per beam per step |
| `--length-penalty` | 0.7 | beams rank by `logprob / len ** this` |
| `--seed` | none | reproducible run |
| `--verbose` | off | print the distribution behind every character |

Decoding is greedy and has no temperature. A question with one correct answer
gains nothing from variety, and a character-level answer cannot recover from a
wrong character.

As a library:

```python
import random
from jev_lm import build_client, generate

with build_client() as client:
    run = generate(client, "largest planet", random.Random(1), beam_width=3)

print(run.text, run.requests, run.reason)
```

## Layout

```
src/jev_lm/
  model.py              the 28-option vocabulary, the instructions, the state,
                        and score_step() — one request, one distribution
  generation/
    utils.py            generate() dispatch, Run, Step, trace formatting
    greedy.py           argmax and the single-path loop, streamed via on_char
    beam_search.py      Beam and the width-N search
  client.py             build_client() from the environment
  cli.py                the jev-lm command and the question session
experiments/            the A/Bs, with their raw results in results/
archive/                what was tried and cut, with the numbers that closed it
tests/                  offline; no API calls
```

## Results

Beam search beats greedy where a locally-attractive character leads nowhere.
Spelling *washington*, the space after `"capital was"` scores 0.87 against 0.05
for `h`, and greedy has to take it. On `largest planet` across three seeds
(`experiments/ab_beam2.py`):

| arm | correct | requests |
|---|---|---|
| greedy | 2/3 | 8, 8, 4 |
| beam width 3 | 3/3 | 25, 25, 21 |

`archive/README.md` has the rest: the stop-mechanism comparison, why STOP
shaping was dropped, and what temperature sampling cost.

## Tests

```bash
pytest
```

They cover the pure functions — the option set, the ensemble fold, the state, and
beam scoring — and make no API calls.
