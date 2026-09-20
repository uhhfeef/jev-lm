import random
from collections.abc import Callable
from dataclasses import dataclass, field

from typesafe_sdk import TypeSafeClient


@dataclass
class Step:
    index: int
    char: str          # the character appended, or "." when the answer stopped here
    stop_prob: float   # what the STOP option scored on the text before this character
    top5: str
    chosen_prob: float = 0.0  # probability of the character actually taken


@dataclass
class Run:
    text: str
    reason: str                     # stop | max_chars
    requests: int = 0
    steps: list[Step] = field(default_factory=list)


def quoted(text: str) -> str:
    return f'"{text}"'


def top_n(probabilities: dict[str, float], n: int = 5) -> str:
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return " ".join(f"{c}={p:.2f}" for c, p in ranked)


def generate(
    client: TypeSafeClient,
    question: str,
    rng: random.Random,
    *,
    max_chars: int = 60,
    quiet: bool = True,
    window: int = 40,
    ensemble: int = 3,
    beam_width: int = 1,
    length_penalty: float = 0.7,
    on_char: Callable[[str], None] | None = None,
) -> Run:
    """Run the autoregressive loop until the model stops or the budget is spent.

    1. Hold the growing answer in `text`, starting empty.
    2. Ask for the distribution over the 28 options -- one request per step.
    3. Each question is asked `ensemble` times over reshuffled options, averaged.
    4. Take the most probable option -> append its character, or stop on STOP.
    5. Record the step and print it unless `quiet`.
    6. Stop on STOP or after `max_chars` steps.

    `on_char` is called with each character the moment it is appended, so a caller
    can stream the answer out as it is written. It is separate from `quiet`: `quiet`
    controls the diagnostic line carrying the distribution, `on_char` carries only
    the character. Beam search takes no callback -- the leading candidate can change
    after a character would have been printed, so there is nothing honest to stream.

    Above `beam_width` 1 this hands off to `beam_search`.
    """
    from jev_lm.generation.beam_search import beam_search
    from jev_lm.generation.greedy import greedy_search

    if beam_width > 1:
        return beam_search(
            client, question, rng,
            max_chars=max_chars, quiet=quiet, window=window, ensemble=ensemble,
            beam_width=beam_width, length_penalty=length_penalty,
        )
    return greedy_search(
        client, question, rng,
        max_chars=max_chars, quiet=quiet, window=window, ensemble=ensemble,
        on_char=on_char,
    )
