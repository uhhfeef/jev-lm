import argparse
import random

from typesafe_sdk import TypeSafeClient

from .client import build_client
from .generation import Run, generate

PROMPT = "you> "
REPLY = "jev> "
BANNER = "ask a question. blank line to skip, 'exit' or ctrl-c to leave."


def ask(
    client: TypeSafeClient,
    question: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> Run:
    # --verbose interleaves diagnostics, and beam search can still change its
    # leading candidate, so neither streams; both print the finished answer
    stream = args.beam_width == 1 and not args.verbose
    if stream:
        print(REPLY, end="", flush=True)

    run = generate(
        client,
        question,
        rng,
        max_chars=args.max_chars,
        quiet=not args.verbose,
        window=args.window,
        ensemble=args.ensemble,
        beam_width=args.beam_width,
        length_penalty=args.length_penalty,
        on_char=(lambda c: print(c, end="", flush=True)) if stream else None,
    )

    if stream:
        print()
    else:
        print(REPLY + run.text)
    return run


def repl(
    client: TypeSafeClient, rng: random.Random, args: argparse.Namespace
) -> None:
    print(BANNER)
    while True:
        try:
            question = input(f"\n{PROMPT}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return

        try:
            ask(client, question, rng, args)
        except KeyboardInterrupt:
            print("\n[stopped]")
        except Exception as error:  # one bad request, not the end of the session
            print(f"\n[error] {error}")


def main() -> None:
    """Parse the flags, open a client, and answer questions until told to stop.

    1. Read the budget and the search flags. `--question` answers one and exits,
       for scripting; without it the session is interactive.
    2. Seed one Random for option shuffling, shared by every question in the
       session -> `--seed` reproduces a whole session, since decoding is greedy.
    3. Open a TypeSafeClient on TYPESAFE_API_KEY, once, and hold it for as long
       as questions keep coming.
    """
    parser = argparse.ArgumentParser(
        description="Answer a question one character at a time with Jev."
    )
    parser.add_argument(
        "--question", help="answer one question and exit; omit for a session"
    )
    parser.add_argument("--max-chars", type=int, default=60)
    parser.add_argument("--window", type=int, default=40, help="context shown per option")
    parser.add_argument("--seed", type=int, default=None, help="reproducible run")
    parser.add_argument(
        "--verbose", action="store_true",
        help="print the distribution behind every character",
    )
    parser.add_argument(
        "--ensemble", type=int, default=3,
        help="shuffled option orderings averaged per step; free in requests",
    )
    parser.add_argument(
        "--beam-width", type=int, default=1,
        help="candidates kept alive; >1 costs one request per beam per step",
    )
    parser.add_argument(
        "--length-penalty", type=float, default=0.7,
        help="beams rank by logprob / len ^ this; 0.7 is the measured default",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with build_client() as client:
        if args.question:
            ask(client, args.question, rng, args)
        else:
            repl(client, rng, args)


if __name__ == "__main__":
    main()
