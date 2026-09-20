import random

from typesafe_sdk import Choice, ChoiceAnswer, JSONValue, TypeSafeClient

MODEL = "jev-latest"

LETTERS = "abcdefghijklmnopqrstuvwxyz"
# the space option needs a name, because a bare " " is invisible both as an option
# label and in the printed trace; the answer itself holds a real space
SPACE_KEY = "space"
SPACE = " "
STOP = "."
# the 28th option: the answer with nothing added to it.
STOP_KEY = "STOP"

TASK = (
    "A correct English answer to `question` is being written out one character at a "
    "time, left to right. `answer_so_far` is exactly the text written so far, "
    "character for character. Exactly one character is appended to it next, verbatim. "
    "The writer is Jev, a model built by TypeSafe."
)

NEXT_CHAR_INSTRUCTIONS = (
    "Each option is `answer_so_far` with one more character appended. Pick the option "
    "that reads as the best beginning of a correct, grammatical, factually right "
    "English answer to `question`. Each option is quoted, so the one ending in a gap "
    "before its closing quote is the option that appends a space, finishing the "
    "current word; pick it only when the word before it is spelled out in full. "
    "Exactly one option is `answer_so_far` with nothing added to it: pick that one "
    "only if the answer already answers `question` completely and ends on a finished, "
    "correctly spelled word."
)


def hypotheses(text: str, window: int, rng: random.Random) -> dict[str, str]:
    """Describe each option as the resulting text rather than as a symbol.

    1. Take the last `window` characters of the answer -> a short prefix.
    2. Append each candidate character -> one hypothesis per option.
    3. Offer the space only at a word boundary -> no double spaces.
    4. Add the option that appends nothing -> ending competes with the letters.
    5. Shuffle the order -> Jev's position bias averages out instead of deciding.
    """
    shown = text[-window:]
    options = {c: f'"{shown}{c}"' for c in LETTERS}
    if text and not text.endswith(SPACE):
        options[SPACE_KEY] = f'"{shown}{SPACE}"'
    options[STOP_KEY] = f'"{shown}"'
    keys = list(options)
    rng.shuffle(keys)
    return {k: options[k] for k in keys}


def char_questions(
    text: str, window: int, rng: random.Random, ensemble: int
) -> dict[str, Choice]:
    return {
        f"next{i}": Choice(
            instructions=NEXT_CHAR_INSTRUCTIONS,
            criteria=hypotheses(text, window, rng),
        )
        for i in range(max(1, ensemble))
    }


def average(choices: dict[str, ChoiceAnswer], ensemble: int) -> dict[str, float]:
    n = max(1, ensemble)
    totals: dict[str, float] = {}
    for i in range(n):
        for key, value in choices[f"next{i}"].probabilities.items():
            totals[key] = totals.get(key, 0.0) + float(value) / n
    return totals


def char_state(question: str, text: str) -> dict[str, JSONValue]:
    state: dict[str, JSONValue] = {
        "task": TASK,
        "question": question,
        "answer_so_far": text,
        "current_partial_word": text.rsplit(SPACE, 1)[-1],
    }
    return state


def score_step(
    client: TypeSafeClient,
    question: str,
    text: str,
    window: int,
    rng: random.Random,
    ensemble: int,
) -> dict[str, float]:
    return average(
        client.system_one(
            state=char_state(question, text),
            questions=char_questions(text, window, rng, ensemble),
            model=MODEL,
        ).choices,
        ensemble,
    )
