import argparse
import random

from .client import build_client
from .generation import generate


def main() -> None:
    """Parse the flags, open a client, answer one question, print the answer.

    1. Read the question, the budget and the search flags.
    2. Seed one Random for option shuffling -> `--seed` reproduces a whole run.
    3. Open a TypeSafeClient on TYPESAFE_API_KEY and run `generate`.
    4. Print the finished answer.
    """
    parser = argparse.ArgumentParser(
        description="Answer a question one character at a time with Jev."
    )
    parser.add_argument("--question", required=True, help="what Jev is answering")
    parser.add_argument("--max-chars", type=int, default=60)
    parser.add_argument("--window", type=int, default=40, help="context shown per option")
    parser.add_argument("--seed", type=int, default=None, help="reproducible run")
    parser.add_argument("--quiet", action="store_true", help="only print the result")
    parser.add_argument(
        "--ensemble", type=int, default=3,
        help="shuffled option orderings averaged per step; free in requests",
    )
    parser.add_argument(
        "--beam-width", type=int, default=1,
        help="candidate answers kept alive; >1 costs one request per beam per step",
    )
    parser.add_argument(
        "--length-penalty", type=float, default=0.7,
        help="beams rank by logprob / len ^ this; 0.7 is the measured default",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with build_client() as client:
        run = generate(
            client,
            args.question,
            rng,
            max_chars=args.max_chars,
            quiet=args.quiet,
            window=args.window,
            ensemble=args.ensemble,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
        )

    print(f"\n{run.text}")


if __name__ == "__main__":
    main()
