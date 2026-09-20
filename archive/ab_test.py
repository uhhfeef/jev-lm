"""A/B the three ways the answer can decide to stop, on one fixed question set.

The gate costs a whole extra request per character. This measures whether it buys
anything now that `question` anchors the task -- it was worth it before the anchor
existed, and that is exactly what changed.

    gate       `should_stop` alone, then the character.        2 requests/char
    parallel   both questions in one request.                  1 request/char
    symbol     STOP as a 28th option, no gate.                 1 request/char
    symbol+    the same, discounted by stop_bias and min_steps 1 request/char

`symbol+` is a separate arm because hypothesis options inflate STOP: the unchanged
answer is a valid prefix of every extension of itself. Testing bare `symbol` alone
would reject jevchat's design rather than the real version of it.

Everything runs greedy (`temperature 0`). We already know 0.7 loses the run by
fumbling the first letter, and that noise would swamp the difference between arms.
Greedy is deterministic too, so one run per cell is enough.
"""

import json
import os
import random
import statistics
import sys
import time

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from jev_letter_lm import GATE, PARALLEL, SYMBOL, Run, generate

_ = load_dotenv()

RESULTS = "ab_results.json"

# name -> the generate() keywords that define the arm
ARMS: dict[str, dict[str, object]] = {
    "gate": {"stop_mode": GATE},
    "parallel": {"stop_mode": PARALLEL},
    "symbol": {"stop_mode": SYMBOL, "stop_bias": 1.0, "min_steps": 0},
    "symbol+": {"stop_mode": SYMBOL, "stop_bias": 0.5, "min_steps": 3},
}

# Short factual answers, all spellable in a-z and a space.
QUESTIONS: list[tuple[str, str]] = [
    ("what is the capital of france", "paris"),
    ("what is your name", "jev"),
    ("what color is the sky", "blue"),
    ("what is the largest ocean", "pacific"),
    ("what is two plus two", "four"),
    ("who wrote romeo and juliet", "shakespeare"),
    ("what is the capital of japan", "tokyo"),
    ("how many legs does a spider have", "eight"),
]

MAX_CHARS = 25
WINDOW = 40
SEED = 0


def separation(run: Run, expected: str) -> tuple[list[float], list[float]]:
    """Split the run's stop probabilities by whether the answer was done yet.

    Each step's `stop_prob` was judged on the text *before* that step's character,
    so the prefix is rebuilt from the preceding steps. The gap between the two means
    is the calibration number: it says whether the stop signal actually knows the
    difference between a finished answer and an unfinished one.
    """
    unfinished: list[float] = []
    complete: list[float] = []
    prefix = ""
    for step in run.steps:
        (complete if expected in prefix else unfinished).append(step.stop_prob)
        prefix += step.char
    return unfinished, complete


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def run_cell(client: TypeSafeClient, question: str, arm: dict[str, object]) -> Run:
    return generate(
        client,
        question,
        "",            # no seeded opening: the answer is the model's whole job
        MAX_CHARS,
        True,          # quiet; the harness prints its own line per cell
        0.0,           # greedy, fixed across every arm
        5,
        WINDOW,
        random.Random(SEED),
        **arm,         # pyright: ignore[reportArgumentType]
    )


def main() -> None:
    """Run every arm against every question, print a summary, save the raw steps."""
    records: list[dict[str, object]] = []
    started = time.monotonic()

    with TypeSafeClient(
        api_key=os.environ["TYPESAFE_API_KEY"],
        base_url=os.environ.get("TYPESAFE_ENDPOINT"),
        timeout=60.0,
    ) as client:
        for arm_name, arm in ARMS.items():
            for question, expected in QUESTIONS:
                run = run_cell(client, question, arm)
                unfinished, complete = separation(run, expected)
                record = {
                    "arm": arm_name,
                    "question": question,
                    "expected": expected,
                    "text": run.text,
                    "correct": expected in run.text,
                    "reason": run.reason,
                    "length": len(run.text),
                    "requests": run.requests,
                    "p_stop_unfinished": mean(unfinished),
                    "p_stop_complete": mean(complete),
                    "steps": [
                        {"i": s.index, "char": s.char, "stop": s.stop_prob, "top5": s.top5}
                        for s in run.steps
                    ],
                }
                records.append(record)
                flag = "OK " if record["correct"] else "   "
                print(
                    f"{flag}{arm_name:<9} {expected:<12} "
                    f"{run.reason:<9} {run.requests:>4} req  {run.text!r}",
                    flush=True,
                )
                # written every cell, so a cell is never paid for twice
                with open(RESULTS, "w", encoding="utf-8") as handle:
                    json.dump(records, handle, indent=1)

    print(f"\n{len(records)} cells in {time.monotonic() - started:.0f}s -> {RESULTS}\n")
    summarise(records)


def summarise(records: list[dict[str, object]]) -> None:
    """One row per arm: accuracy, how it ended, what it cost, and the separation."""
    n = len(QUESTIONS)
    print(f"{'arm':<9} {'correct':>8} {'stopped':>8} {'req/ans':>8} "
          f"{'len':>6} {'P(stop) unfin':>14} {'complete':>9} {'sep':>7}")
    print("-" * 76)
    for arm_name in ARMS:
        rows = [r for r in records if r["arm"] == arm_name]
        if not rows:
            continue
        unfin = mean([r["p_stop_unfinished"] for r in rows
                      if r["p_stop_unfinished"] == r["p_stop_unfinished"]])
        comp = mean([r["p_stop_complete"] for r in rows
                     if r["p_stop_complete"] == r["p_stop_complete"]])
        print(
            f"{arm_name:<9} "
            f"{sum(bool(r['correct']) for r in rows):>5}/{n} "
            f"{sum(r['reason'] == 'stop' for r in rows):>5}/{n} "
            f"{mean([float(r['requests']) for r in rows]):>8.1f} "
            f"{mean([float(r['length']) for r in rows]):>6.1f} "
            f"{unfin:>14.2f} {comp:>9.2f} {comp - unfin:>+7.2f}"
        )


if __name__ == "__main__":
    sys.exit(main())
