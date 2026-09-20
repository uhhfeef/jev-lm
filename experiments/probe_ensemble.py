"""Is the near-tie after "j" position bias, or does the model genuinely not know?

Asked for the largest planet the loop writes `j`, and then the next character is a
coin flip -- `u=0.31 a=0.25 o=0.17` on one run, `o=0.26 a=0.18 u=0.18` on the next.
`u` gives *jupiter* and the remaining characters are near-forced; `a` and `o` give
`janos` and `jovis`. So one near-tie decides the whole answer.

Two explanations needing opposite fixes:

  noise  Jev favours whichever options come first, and we reshuffle every run. Then
         averaging more orderings pulls `u` clear, and raising `--ensemble` is the
         whole fix.
  real   The model does not know. More averaging changes nothing and only lookahead
         helps.

One frozen prefix, four ensemble sizes, one request each: the orderings ride along
as parallel questions. ~10 requests total, no generation.
"""

import random
import statistics

from jev_lm import build_client

from jev_lm import (
    MODEL,
    average,
    char_questions,
    char_state,
)

QUESTION = "largest planet"
PREFIX = "j"          # the step where the answer is decided
WINDOW = 40
SIZES = (1, 3, 6, 12)
WATCH = ("u", "a", "o", "e", "i")   # u is jupiter; the rest are the wrong turns


def main() -> None:
    """Score the same step at each ensemble size and print how the field moves."""
    rows: dict[int, dict[str, float]] = {}

    with build_client() as client:
        for size in SIZES:
            # a fresh rng per size, so each size is an independent set of shuffles
            rng = random.Random(1234)
            rows[size] = average(
                client.system_one(
                    state=char_state(QUESTION, PREFIX),
                    questions=char_questions(PREFIX, WINDOW, rng, size),
                    model=MODEL,
                ).choices,
                size,
            )
            print(".", end="", flush=True)
    print()

    print(f"\nquestion {QUESTION!r}, answer_so_far {PREFIX!r}\n")
    print(f"{'ensemble':>9} " + " ".join(f"{c:>7}" for c in WATCH)
          + f" {'margin':>8} {'spread':>8}")
    print("-" * 62)
    for size in SIZES:
        probs = rows[size]
        watched = [probs.get(c, 0.0) for c in WATCH]
        ranked = sorted(probs.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        print(f"{size:>9} " + " ".join(f"{v:>7.3f}" for v in watched)
              + f" {margin:>+8.3f} {statistics.pstdev(watched):>8.3f}")

    first, last = rows[SIZES[0]], rows[SIZES[-1]]
    winner = max(last, key=last.__getitem__)
    print(f"\ntop option at ensemble {SIZES[0]}: {max(first, key=first.__getitem__)!r}"
          f"   at ensemble {SIZES[-1]}: {winner!r}")
    print(f"p(u) {first.get('u', 0.0):.3f} -> {last.get('u', 0.0):.3f}; "
          "climbing means position bias, flat means the model does not know")


if __name__ == "__main__":
    main()
