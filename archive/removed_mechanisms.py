"""Mechanisms tried in `jev_letter_lm.py` and cut, kept here with their numbers.

Nothing in the live loop imports this. It is here so the next person to ask "why
not gate stopping behind its own question?" or "where did temperature go?" can
read the code that answered it next to the measurement that closed it.
`README.md` in this directory is the summary; this file is the source.

It imports the pieces that survived -- `TASK`, `IDENTITY_FULL`, `ABOUT_YOU_WORDS`,
`char_state`, `average` -- so the archived code stays readable rather than
becoming a wall of undefined names. `ab_test.py` and `probe_stop.py`, also in this
directory, are the harnesses that produced the numbers; they still import the old
names from `jev_letter_lm` and would need repointing here to run again.
"""

import random
import re

from typesafe_sdk import Choice, JSONValue, Noul, NoulCriteria, TypeSafeClient

from jev_lm.model import (
    ABOUT_YOU_WORDS,
    IDENTITY_FULL,
    LETTERS,
    MODEL,
    SPACE,
    SPACE_KEY,
    STOP_KEY,
    TASK,
    average,
    char_state,
)

# ---------------------------------------------------------------------------
# 1. Stopping as a separate question: `--stop-mode gate` and `--stop-mode parallel`
# ---------------------------------------------------------------------------
# Ending the answer was judged by its own boolean question instead of competing
# with the letters. `gate` asked it first, in a request of its own, and only paid
# for the character if it did not fire -- two requests per character. `parallel`
# batched it alongside the character question -- one request, but the gate still
# could not see how the alphabet was leaning.
#
# `probe_stop.py` scored all three over eight hand-labelled prefixes:
#
#     mechanism   unfinished    complete     separation   requests/char
#     gate        0.01-0.14     0.70-0.89      +0.76           2
#     parallel    0.01-0.16     0.65-0.90      +0.75           1
#     symbol      0.00-0.03     0.92-0.99      +0.95           1
#
# Reading STOP off the character distribution wins on both axes, so both of these
# went. `symbol` is now the whole of the live file, which is why it no longer has
# a `--stop-mode` flag to select between them.

STOP_THRESHOLD = 0.5  # `should_stop` at or above this ended the answer

GATE, PARALLEL, SYMBOL = "gate", "parallel", "symbol"
STOP_MODES = (GATE, PARALLEL, SYMBOL)

# With ending judged elsewhere, it had to be kept out of the character question
# explicitly. The live file carries only the other variant -- the one that
# describes the STOP option -- so `NEXT_CHAR_BODY` is no longer split in two.
NEXT_CHAR_BODY = (
    "Each option is `answer_so_far` with one more character appended. Pick the option "
    "that reads as the best beginning of a correct, grammatical, factually right "
    "English answer to `question`. Each option is quoted, so the one ending in a gap "
    "before its closing quote is the option that appends a space, finishing the "
    "current word; pick it only when the word before it is spelled out in full. If "
    "`question` asks about you, answer it with the truth stated in `about_you`."
)

NEXT_CHAR_INSTRUCTIONS = NEXT_CHAR_BODY + (
    " Assume the answer is continuing: whether it should end instead is decided "
    "separately, so never treat ending it as one of your options here."
)

SHOULD_STOP = Noul(
    instructions=(
        "Should the answer in `answer_so_far` end here, with a full stop, instead of "
        "continuing with another character? Judge only whether it has completely "
        "answered `question`. `characters_generated` against `max_characters` shows "
        "how much of the budget is left."
    ),
    criteria=NoulCriteria(
        true=(
            "`answer_so_far` completely answers `question`, is a grammatical "
            "statement, and `current_partial_word` is a whole correctly spelled word, "
            "so a full stop can attach directly to it."
        ),
        false=(
            "The answer is still mid-thought, has not yet answered `question`, or "
            "`current_partial_word` is empty or a half-spelled fragment, so it needs "
            "at least one more character."
        ),
    ),
)


def stop_state(
    question: str, text: str, step: int, max_chars: int, identity: str | None
) -> dict[str, JSONValue]:
    """The state for `should_stop`: the answer plus what the gate needs to judge it.

    Differs from the surviving `char_state` only in the two budget fields, which
    were there so the gate could lean toward stopping as the budget ran down.
    """
    state: dict[str, JSONValue] = {
        "task": TASK,
        "question": question,
        "answer_so_far": text,
        "current_partial_word": text.rsplit(SPACE, 1)[-1],
        "characters_generated": step,
        "max_characters": max_chars,
    }
    if identity is not None:
        state["about_you"] = identity
    return state


def letters_only(text: str, window: int, rng: random.Random) -> dict[str, str]:
    """`hypotheses()` without the STOP option, which these two modes never offered.

    The live version always includes it, so its `include_stop` parameter is gone.
    """
    shown = text[-window:]
    options = {c: f'"{shown}{c}"' for c in LETTERS}
    if text and not text.endswith(SPACE):
        options[SPACE_KEY] = f'"{shown}{SPACE}"'
    keys = list(options)
    rng.shuffle(keys)
    return {k: options[k] for k in keys}


def gate_questions(
    text: str, window: int, rng: random.Random, ensemble: int
) -> dict[str, Choice]:
    """The character question for these modes: no STOP option, and told to ignore it."""
    return {
        f"next{i}": Choice(
            instructions=NEXT_CHAR_INSTRUCTIONS,
            criteria=letters_only(text, window, rng),
        )
        for i in range(max(1, ensemble))
    }


def score_step_gate(
    client: TypeSafeClient,
    question: str,
    text: str,
    step: int,
    max_chars: int,
    window: int,
    rng: random.Random,
    min_steps: int,
    ensemble: int,
    identity: str | None,
) -> tuple[dict[str, float], float, bool, int]:
    """`--stop-mode gate`: two requests per character, the second one conditional.

    The stop judgment goes first, in a request of its own: it is a gate on the
    alphabet, not a participant in it, so it should not see the alphabet. Returns
    `(probabilities, stop_prob, gate_fired, requests)` -- the four fields of the
    `Scored` dataclass the live file no longer needs, since without a gate every
    step costs exactly one request and nothing can fire.
    """
    stop_prob = client.system_one(
        state=stop_state(question, text, step, max_chars, identity),
        questions={"should_stop": SHOULD_STOP},
        model=MODEL,
    ).nouls["should_stop"].noul
    if stop_prob >= STOP_THRESHOLD and step >= min_steps:
        return {}, stop_prob, True, 1
    probabilities = average(
        client.system_one(
            state=char_state(question, text, identity),
            questions=gate_questions(text, window, rng, ensemble),
            model=MODEL,
        ).choices,
        ensemble,
    )
    return probabilities, stop_prob, False, 2


def score_step_parallel(
    client: TypeSafeClient,
    question: str,
    text: str,
    step: int,
    max_chars: int,
    window: int,
    rng: random.Random,
    min_steps: int,
    ensemble: int,
    identity: str | None,
) -> tuple[dict[str, float], float, bool, int]:
    """`--stop-mode parallel`: the same gate, batched into the character request."""
    response = client.system_one(
        state=stop_state(question, text, step, max_chars, identity),
        questions={
            "should_stop": SHOULD_STOP,
            **gate_questions(text, window, rng, ensemble),
        },
        model=MODEL,
    )
    stop_prob = response.nouls["should_stop"].noul
    fired = stop_prob >= STOP_THRESHOLD and step >= min_steps
    return average(response.choices, ensemble), stop_prob, fired, 1


# ---------------------------------------------------------------------------
# 2. Shaping the STOP option: `--stop-bias` and `--min-steps`
# ---------------------------------------------------------------------------
# The worry was that hypothesis options inflate STOP: the unchanged answer is a
# valid prefix of every extension of itself, so it might collect mass on that
# technicality alone. `--stop-bias 0.5 --min-steps 3` was the correction, and
# `probe_stop.py` scored it as the `symbol+` arm:
#
#     prefix                             symbol   symbol+
#     ""                                  0.010     0.005
#     "par"                               0.000     0.000
#     "paris is"                          0.027     0.014
#     "paris"                             0.950     0.905
#     "the capital of france is paris"    0.977     0.954
#     "je"                                0.000     0.000
#     "jev"                               0.917     0.846
#     "my name is jev"                    0.987     0.974
#
# The correction is real but points the wrong way. The unfinished side was already
# at 0.00-0.03, so there was no inflation left to discount, and every complete
# prefix lost 4-7 points of confidence for nothing. The empty answer scores 0.01
# unshaped, so `--min-steps` was guarding against something that does not happen.


def shape_stop(
    probabilities: dict[str, float], stop_bias: float, ban_stop: bool
) -> dict[str, float]:
    """Bias or ban the STOP option, then renormalise."""
    out = dict(probabilities)
    if ban_stop:
        _ = out.pop(STOP_KEY, None)
    elif stop_bias != 1.0 and STOP_KEY in out:
        out[STOP_KEY] *= max(0.0, stop_bias)
    total = sum(out.values())
    if total <= 0:  # nothing survived; fall back rather than divide by zero
        return dict(probabilities)
    return {k: v / total for k, v in out.items()}


# ---------------------------------------------------------------------------
# 3. Sampling: `--temperature`, `--top-k`, `--top-p`
# ---------------------------------------------------------------------------
# A question with one correct answer has nothing to gain from variety, and a
# character-level answer cannot recover from a wrong character, so every degree
# above 0 is a fresh chance to lose with no upside. Asked for the largest planet,
# Jev ranked `t=0.31` over `e=0.22` at "jupi" -- preferring the right letter --
# and sampling at 0.4 still drew `e` 28% of the time, which is `jupiesne.` instead
# of `jupiter.`. So `--temperature` defaulted to 0, every call site left it there,
# and `top_k` and `top_p` sat below a branch that never ran. The live loop calls
# `argmax` directly.


def sample(
    probabilities: dict[str, float],
    temperature: float,
    top_k: int,
    top_p: float,
    rng: random.Random,
) -> str:
    """Draw one character from the distribution, or take the argmax at zero.

    1. `temperature` of 0 -> greedy decoding, worth being able to reach exactly.
    2. Reweight by `p ** (1 / temperature)` and renormalise -> a sharper
       distribution below 1.0, a flatter one above.
    3. `top_k` keeps that many characters, or all of them at 0 -> a fixed cut.
    4. `top_p` keeps the smallest set whose mass reaches it -> an adaptive cut, wide
       where Jev is unsure and narrow where it is confident, which a fixed `top_k`
       cannot be.

    The order matters: temperature first, then the filters, so the cut is made on
    the distribution actually being drawn from.
    """
    if temperature <= 0:
        return max(probabilities, key=lambda c: probabilities[c])

    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    weights = [p ** (1 / temperature) for _, p in ranked]
    total = sum(weights)
    if total <= 0:  # a fully flat-zero distribution has nothing to sample
        return ranked[0][0]
    # `** (1 / temperature)` is monotonic over non-negative p, so order is preserved
    scaled = [(c, w / total) for (c, _), w in zip(ranked, weights)]

    if top_k > 0:
        scaled = scaled[:top_k]

    if top_p < 1.0:
        kept: list[tuple[str, float]] = []
        mass = 0.0
        for entry in scaled:
            kept.append(entry)
            mass += entry[1]
            if mass >= top_p:
                break
        scaled = kept

    return rng.choices([c for c, _ in scaled], weights=[w for _, w in scaled])[0]


# ---------------------------------------------------------------------------
# 4. Identity modes: `--identity full` and `--identity unspelled`
# ---------------------------------------------------------------------------
# `about_you` tells Jev who it is, so "what is your name" can be answered
# truthfully. Sending the spelling on every request taught the model to write it:
# asked for the largest planet it answered `jev pa .`, taking `e=0.35` over
# `u=0.15` at the second character of what should have been *jupiter*.
# `unspelled` dropped the spelling everywhere; `gated` sends the spelled identity
# only when the question is about the model, and won. `gated` is now the only
# behaviour, so `identity_for` takes a question and nothing else.

IDENTITY_PLAIN = (
    "You are Jev, a System One model built by TypeSafe. Here you are acting "
    "as a character-level language model, emitting one character at a time."
)

FULL, UNSPELLED, GATED = "full", "unspelled", "gated"
IDENTITY_MODES = (FULL, UNSPELLED, GATED)


def identity_for(question: str, mode: str) -> str | None:
    """The `about_you` to send with this question, or None to leave it out."""
    if mode == FULL:
        return IDENTITY_FULL
    if mode == UNSPELLED:
        return IDENTITY_PLAIN
    words = set(re.findall(r"[a-z]+", question.lower()))
    return IDENTITY_FULL if words & ABOUT_YOU_WORDS else None


# ---------------------------------------------------------------------------
# 5. The seeded opening: `--prompt`
# ---------------------------------------------------------------------------
# `generate(prompt=...)` let the answer start from a given string rather than "".
# Nothing ever used it -- every call site in the repo passed "" -- and having the
# harness write the opening would undercut the thing being tested, which is
# whether Jev can produce one unaided. The loop now always starts empty.
