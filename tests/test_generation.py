import random

from jev_lm.generation import greedy
from jev_lm.model import STOP_KEY


def scripted(*distributions):
    """A `score_step` that hands back one fixed distribution per call."""
    calls = iter(distributions)
    return lambda *args, **kwargs: next(calls)


def test_on_char_sees_exactly_the_characters_of_the_answer(monkeypatch):
    monkeypatch.setattr(
        greedy, "score_step", scripted({"h": 1.0}, {"i": 1.0}, {STOP_KEY: 1.0})
    )
    seen: list[str] = []

    run = greedy.greedy_search(
        client=None,
        question="q",
        rng=random.Random(0),
        max_chars=5,
        quiet=True,
        window=40,
        ensemble=1,
        on_char=seen.append,
    )

    assert "".join(seen) == run.text
    assert run.text == "hi."


def test_greedy_runs_without_a_callback(monkeypatch):
    monkeypatch.setattr(greedy, "score_step", scripted({"a": 1.0}, {STOP_KEY: 1.0}))

    run = greedy.greedy_search(
        client=None,
        question="q",
        rng=random.Random(0),
        max_chars=5,
        quiet=True,
        window=40,
        ensemble=1,
    )

    assert run.text == "a."
    assert run.reason == "stop"
