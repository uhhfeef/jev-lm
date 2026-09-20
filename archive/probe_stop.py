"""Score the three stop mechanisms on fixed prefixes, without generating anything.

The arms differ in one thing: how good the "is this answer finished?" signal is.
Generating whole answers to find that out cost ~1,000 requests and scored every arm
on option-order luck, because at temperature 0 the shuffle seed alone decided
whether the france answer came out `paris.` or `cap t al e s a s a r a a`.

So: hand-written prefixes of known completeness, one request per prefix per arm,
nothing autoregressive, nothing that compounds. 24 requests total.

`symbol+` costs no requests of its own -- `stop_bias` is a multiplier applied to the
raw STOP probability after the fact, so it is derived arithmetically from `symbol`.
"""

import json
import os
import random
import statistics

from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

from jev_letter_lm import (
    MODEL,
    NEXT_CHAR_INSTRUCTIONS,
    NEXT_CHAR_INSTRUCTIONS_STOP,
    SHOULD_STOP,
    STOP_KEY,
    IDENTITY_FULL,
    STOP_THRESHOLD,
    char_state,
    hypotheses,
    stop_state,
)

_ = load_dotenv()

RESULTS = "probe_results.json"
WINDOW = 40
BUDGET = 25       # the `max_characters` a real run would be working against
ORDERINGS = 3     # shuffled criteria orderings, batched into one request
STOP_BIAS = 0.5   # the `symbol+` correction, applied arithmetically

FRANCE = "what is the capital of france"
NAME = "what is your name"

# (question, answer_so_far, is the answer actually complete)
# #3 is the discriminator: a whole, correctly spelled word that does not yet answer
# the question. A signal keying on "is the word finished" passes it; one keying on
# "is the question answered" fails it.
PROBES: list[tuple[str, str, bool]] = [
    (FRANCE, "", False),
    (FRANCE, "par", False),
    (FRANCE, "paris is", False),
    (FRANCE, "paris", True),
    (FRANCE, "the capital of france is paris", True),
    (NAME, "je", False),
    (NAME, "jev", True),
    (NAME, "my name is jev", True),
]


def ask_gate(client: TypeSafeClient, question: str, text: str) -> float:
    """The Noul on its own, exactly as `--stop-mode gate` asks it."""
    return client.system_one(
        state=stop_state(question, text, len(text), BUDGET, IDENTITY_FULL),
        questions={"should_stop": SHOULD_STOP},
        model=MODEL,
    ).nouls["should_stop"].noul


def ask_parallel(client: TypeSafeClient, question: str, text: str) -> float:
    """The same Noul, but with the alphabet in the request beside it."""
    rng = random.Random(0)
    return client.system_one(
        state=stop_state(question, text, len(text), BUDGET, IDENTITY_FULL),
        questions={
            "should_stop": SHOULD_STOP,
            "next_char": Choice(
                instructions=NEXT_CHAR_INSTRUCTIONS,
                criteria=hypotheses(text, WINDOW, rng),
            ),
        },
        model=MODEL,
    ).nouls["should_stop"].noul


def ask_symbol(client: TypeSafeClient, question: str, text: str) -> float:
    """STOP as a 28th option, averaged over `ORDERINGS` shuffles in one request.

    Jev answers the questions in a request independently and latency is near-flat in
    their number, so the orderings cost one request between them. Averaging cancels
    the position bias that made the generation-based test unreadable.
    """
    rng = random.Random(0)
    questions = {
        f"next{i}": Choice(
            instructions=NEXT_CHAR_INSTRUCTIONS_STOP,
            criteria=hypotheses(text, WINDOW, rng, include_stop=True),
        )
        for i in range(ORDERINGS)
    }
    choices = client.system_one(
        state=char_state(question, text, IDENTITY_FULL), questions=questions, model=MODEL
    ).choices
    return statistics.fmean(
        choices[f"next{i}"].probabilities.get(STOP_KEY, 0.0) for i in range(ORDERINGS)
    )


def biased(p: float, bias: float) -> float:
    """P(stop) after multiplying STOP by `bias` and renormalising the distribution."""
    return bias * p / (bias * p + (1.0 - p)) if 0.0 <= p < 1.0 else p


def main() -> None:
    """Score every probe with every arm, print the tables, save the raw numbers."""
    arms = {"gate": ask_gate, "parallel": ask_parallel, "symbol": ask_symbol}
    scores: dict[str, list[float]] = {name: [] for name in arms}

    with TypeSafeClient(
        api_key=os.environ["TYPESAFE_API_KEY"],
        base_url=os.environ.get("TYPESAFE_ENDPOINT"),
        timeout=60.0,
    ) as client:
        for name, ask in arms.items():
            for question, text, _complete in PROBES:
                scores[name].append(ask(client, question, text))
                print(".", end="", flush=True)
        print()

    scores["symbol+"] = [biased(p, STOP_BIAS) for p in scores["symbol"]]

    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(
            {"probes": [{"question": q, "text": t, "complete": c} for q, t, c in PROBES],
             "scores": scores},
            handle, indent=1,
        )

    report(scores)


def report(scores: dict[str, list[float]]) -> None:
    """Per-probe probabilities, then the three numbers that decide it."""
    names = list(scores)
    print(f"\n{'answer_so_far':<32} {'done':>5} " + " ".join(f"{n:>9}" for n in names))
    print("-" * (39 + 10 * len(names)))
    for i, (_q, text, complete) in enumerate(PROBES):
        row = " ".join(f"{scores[n][i]:>9.2f}" for n in names)
        print(f"{text!r:<32} {'yes' if complete else 'no':>5} {row}")

    print(f"\n{'arm':<10} {'threshold acc':>14} {'unfinished':>11} {'complete':>9} "
          f"{'sep':>7} {'probe 3':>8}")
    print("-" * 64)
    for name in names:
        values = scores[name]
        unfin = [v for v, (_q, _t, c) in zip(values, PROBES) if not c]
        comp = [v for v, (_q, _t, c) in zip(values, PROBES) if c]
        correct = sum(
            (v >= STOP_THRESHOLD) == c for v, (_q, _t, c) in zip(values, PROBES)
        )
        print(f"{name:<10} {correct:>11}/{len(PROBES)} "
              f"{statistics.fmean(unfin):>11.2f} {statistics.fmean(comp):>9.2f} "
              f"{statistics.fmean(comp) - statistics.fmean(unfin):>+7.2f} "
              f"{values[2]:>8.2f}")
    print(f"\nthreshold {STOP_THRESHOLD}; probe 3 is 'paris is' -- whole word, "
          "unfinished answer, so lower is better")


if __name__ == "__main__":
    main()
