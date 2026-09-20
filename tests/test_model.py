import random

from jev_lm.generation import Beam, Step
from jev_lm.model import LETTERS, SPACE_KEY, STOP_KEY, average, char_state, hypotheses


def test_hypotheses_offers_every_letter_plus_space_and_stop():
    options = hypotheses("jup", window=40, rng=random.Random(0))
    assert set(options) == {*LETTERS, SPACE_KEY, STOP_KEY}
    assert len(options) == 28


def test_hypotheses_withholds_space_at_the_start_and_after_a_space():
    assert SPACE_KEY not in hypotheses("", window=40, rng=random.Random(0))
    assert SPACE_KEY not in hypotheses("paris ", window=40, rng=random.Random(0))
    assert SPACE_KEY in hypotheses("paris", window=40, rng=random.Random(0))


def test_stop_hypothesis_is_the_answer_unchanged():
    options = hypotheses("jev", window=40, rng=random.Random(0))
    assert options[STOP_KEY] == '"jev"'
    assert options["a"] == '"jeva"'
    assert options[SPACE_KEY] == '"jev "'


def test_hypotheses_truncates_to_the_window():
    options = hypotheses("abcdefghij", window=3, rng=random.Random(0))
    assert options["z"] == '"hijz"'


def test_hypotheses_shuffles_the_order_without_changing_the_keys():
    a = hypotheses("jev", window=40, rng=random.Random(1))
    b = hypotheses("jev", window=40, rng=random.Random(7))
    assert set(a) == set(b)
    assert list(a) != list(b)


class _Answer:
    def __init__(self, probabilities):
        self.probabilities = probabilities


def test_average_folds_the_ensemble_key_by_key():
    choices = {
        "next0": _Answer({"a": 1.0, "b": 0.0}),
        "next1": _Answer({"a": 0.0, "b": 1.0}),
    }
    assert average(choices, 2) == {"a": 0.5, "b": 0.5}


def test_average_of_one_ordering_is_that_ordering():
    choices = {"next0": _Answer({"a": 0.7, "b": 0.3})}
    folded = average(choices, 1)
    assert folded == {"a": 0.7, "b": 0.3}
    assert abs(sum(folded.values()) - 1.0) < 1e-9


def test_char_state_names_the_partial_word():
    state = char_state("capital of united states", "washing")
    assert state["current_partial_word"] == "washing"
    assert char_state("q", "capital was")["current_partial_word"] == "was"
    assert char_state("q", "paris ")["current_partial_word"] == ""


def test_empty_beam_scores_zero():
    assert Beam(text="").score(0.7) == 0.0


def test_length_penalty_sits_between_raw_logprob_and_the_mean():
    short = Beam(
        text="ab", logprob=-2.0, steps=[Step(0, "a", 0.0, ""), Step(1, "b", 0.0, "")]
    )
    long = Beam(
        text="abcd",
        logprob=-4.0,
        steps=[Step(i, c, 0.0, "") for i, c in enumerate("abcd")],
    )
    # both spend the same log probability per character
    assert long.score(1.0) == short.score(1.0)     # full mean: a tie
    assert long.score(0.0) < short.score(0.0)      # raw logprob: length alone loses
    assert short.score(0.7) > long.score(0.7)      # 0.7 still favours the shorter
    assert long.score(0.7) > long.score(0.0)       # but by less than no division does


def test_beam_extend_emits_the_right_character_per_key():
    base = Beam(text="jev")
    assert base.extend("a", 0.5, 0.0, "").text == "jeva"
    assert base.extend(SPACE_KEY, 0.5, 0.0, "").text == "jev "
    stopped = base.extend(STOP_KEY, 0.9, 0.9, "")
    assert stopped.text == "jev." and stopped.finished
