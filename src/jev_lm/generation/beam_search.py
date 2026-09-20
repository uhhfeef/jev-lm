import math
import random
from dataclasses import dataclass, field

from typesafe_sdk import TypeSafeClient

from jev_lm.model import SPACE, SPACE_KEY, STOP, STOP_KEY, score_step

from .utils import Run, Step, quoted, top_n


@dataclass
class Beam:
    text: str
    logprob: float = 0.0
    finished: bool = False
    steps: list[Step] = field(default_factory=list)

    def score(self, length_penalty: float) -> float:
        if not self.steps:
            return 0.0
        return self.logprob / (len(self.steps) ** length_penalty)

    def extend(
        self, key: str, probability: float, stop_prob: float, top5: str
    ) -> "Beam":
        emit = STOP if key == STOP_KEY else (SPACE if key == SPACE_KEY else key)
        return Beam(
            text=self.text + emit,
            logprob=self.logprob + math.log(max(probability, 1e-12)),
            finished=key == STOP_KEY,
            steps=[
                *self.steps,
                Step(len(self.steps), emit, stop_prob, top5, probability),
            ],
        )


def beam_search(
    client: TypeSafeClient,
    question: str,
    rng: random.Random,
    *,
    max_chars: int,
    quiet: bool,
    window: int,
    ensemble: int,
    beam_width: int,
    length_penalty: float,
) -> Run:
    beams = [Beam(text="")]
    requests = 0

    for _step in range(max_chars):
        live = [b for b in beams if not b.finished]
        if not live:
            break

        candidates = [b for b in beams if b.finished]
        for beam in live:
            probabilities = score_step(
                client, question, beam.text, window, rng, ensemble
            )
            requests += 1
            stop_prob = probabilities.get(STOP_KEY, 0.0)
            top5 = top_n(probabilities)
            ranked = sorted(
                probabilities.items(), key=lambda kv: kv[1], reverse=True
            )[:beam_width]
            for key, probability in ranked:
                candidates.append(beam.extend(key, probability, stop_prob, top5))

        beams = sorted(
            candidates, key=lambda b: b.score(length_penalty), reverse=True
        )[:beam_width]

        if not quiet:
            leader = beams[0]
            print(f"{quoted(leader.text):<46} | score={leader.score(length_penalty):+.3f}"
                  f" | {len(beams)} beams")

    best = max(beams, key=lambda b: b.score(length_penalty))
    return Run(
        text=best.text,
        reason="stop" if best.finished else "max_chars",
        requests=requests,
        steps=best.steps,
    )
