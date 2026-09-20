import random
from collections.abc import Callable

from typesafe_sdk import TypeSafeClient

from jev_lm.model import SPACE, SPACE_KEY, STOP, STOP_KEY, score_step

from .utils import Run, Step, quoted, top_n


def argmax(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda c: probabilities[c])


def greedy_search(
    client: TypeSafeClient,
    question: str,
    rng: random.Random,
    *,
    max_chars: int,
    quiet: bool,
    window: int,
    ensemble: int,
    on_char: Callable[[str], None] | None = None,
) -> Run:
    text = ""
    run = Run(text=text, reason="max_chars")

    for step in range(max_chars):
        probabilities = score_step(
            client, question, text, window, rng, ensemble
        )
        run.requests += 1

        key = argmax(probabilities)
        stop_prob = probabilities.get(STOP_KEY, 0.0)
        shown = top_n(probabilities)
        emit = STOP if key == STOP_KEY else (SPACE if key == SPACE_KEY else key)

        text += emit
        run.steps.append(
            Step(step, emit, stop_prob, shown, probabilities.get(key, 0.0))
        )
        if on_char is not None:
            on_char(emit)
        if not quiet:
            print(f"{quoted(text):<46} | stop={stop_prob:.2f} | {shown}")

        if key == STOP_KEY:
            run.reason = "stop"
            break

    run.text = text
    return run
