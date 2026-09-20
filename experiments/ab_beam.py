"""A/B beam search against the single-path loop, on three questions.

The single-path loop commits to one character per step and can never revisit it, so
a locally-attractive character that leads nowhere is fatal -- spelling *washington*,
the space after "capital was" scored 0.87 against 0.05 for 'h'. Beam search keeps
both branches alive and lets the next few characters decide.

Both arms are greedy. Above width 1 beams are ranked by probability rather than
drawn from, so sampling in the baseline would compare two different mechanisms.

Budget: one request per live beam per step, so the beam arm is ~3x. Results are
written after every cell, so nothing is ever paid for twice.
"""

import json
import math
import random
import time

from typesafe_sdk import TypeSafeClient

from jev_lm import Run, build_client, generate

RESULTS = "results/beam_results.json"
MAX_CHARS = 24
WINDOW = 40
ENSEMBLE = 3

ARMS: dict[str, int] = {"greedy": 1, "beam3": 3}

# (question, expected, why it is here)
QUESTIONS: list[tuple[str, str, str]] = [
    ("capital of united states", "washington", "the 'was' prefix trap"),
    ("capital of france", "paris", "short-answer control"),
    ("largest planet", "jupiter", "medium length, no trap"),
]


def run_cell(client: TypeSafeClient, question: str, width: int) -> Run:
    return generate(
        client,
        question,
        random.Random(0),
        max_chars=MAX_CHARS,
        quiet=True,    # the harness prints one line per cell
        window=WINDOW,
        ensemble=ENSEMBLE,
        beam_width=width,
    )


def main() -> None:
    """Run both arms over every question, save the raw steps, then draw the charts."""
    records: list[dict[str, object]] = []
    started = time.monotonic()
    spent = 0

    with build_client() as client:
        for arm, width in ARMS.items():
            for question, expected, why in QUESTIONS:
                run = run_cell(client, question, width)
                spent += run.requests
                records.append({
                    "arm": arm,
                    "width": width,
                    "question": question,
                    "expected": expected,
                    "why": why,
                    "text": run.text,
                    "correct": expected in run.text,
                    "reason": run.reason,
                    "requests": run.requests,
                    "steps": [
                        {"i": s.index, "char": s.char, "stop": s.stop_prob,
                         "p": s.chosen_prob, "top5": s.top5}
                        for s in run.steps
                    ],
                })
                print(f"{'OK ' if records[-1]['correct'] else '   '}"
                      f"{arm:<8}{expected:<12}{run.reason:<10}{run.requests:>4} req  "
                      f"{run.text!r}   [{spent} total]", flush=True)
                with open(RESULTS, "w", encoding="utf-8") as handle:
                    json.dump(records, handle, indent=1)

    print(f"\n{spent} requests in {time.monotonic() - started:.0f}s -> {RESULTS}")
    summarise(records)
    charts(records)


def summarise(records: list[dict[str, object]]) -> None:
    print(f"\n{'arm':<8} {'correct':>8} {'requests':>9} {'chars':>7} {'mean logp':>10}")
    print("-" * 46)
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        logps = [
            sum(math.log(max(s["p"], 1e-12)) for s in r["steps"]) / max(1, len(r["steps"]))
            for r in rows if r["steps"]
        ]
        print(f"{arm:<8} {sum(bool(r['correct']) for r in rows):>5}/{len(rows)} "
              f"{sum(int(r['requests']) for r in rows):>9} "
              f"{sum(len(str(r['text'])) for r in rows):>7} "
              f"{sum(logps) / max(1, len(logps)):>10.3f}")


def charts(records: list[dict[str, object]]) -> None:
    """Three figures: confidence trajectory, per-step top choice, and cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"greedy": "#c2410c", "beam3": "#1d4ed8"}

    # 1. cumulative mean log probability -- where a run falls off the cliff
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (question, expected, why) in zip(axes, QUESTIONS):
        for arm in ARMS:
            row = next(r for r in records
                       if r["arm"] == arm and r["question"] == question)
            running, total = [], 0.0
            for i, step in enumerate(row["steps"], start=1):
                total += math.log(max(step["p"], 1e-12))
                running.append(total / i)
            ax.plot(range(1, len(running) + 1), running, marker="o", ms=3,
                    color=colours[arm], label=f"{arm} ({'hit' if row['correct'] else 'miss'})")
        ax.set_title(f"{expected}\n{why}", fontsize=9)
        ax.set_xlabel("character")
        ax.axhline(0, lw=0.5, color="#888")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cumulative mean log p")
    fig.suptitle("Confidence per character: higher is a model that knows what it is writing")
    fig.tight_layout()
    fig.savefig("results/chart_logprob.png", dpi=130)

    # 2. probability of the character actually taken, per position
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (question, expected, _why) in zip(axes, QUESTIONS):
        for arm in ARMS:
            row = next(r for r in records
                       if r["arm"] == arm and r["question"] == question)
            ax.plot([s["i"] for s in row["steps"]], [s["p"] for s in row["steps"]],
                    marker="o", ms=3, color=colours[arm], label=arm)
        ax.set_title(expected, fontsize=9)
        ax.set_xlabel("character")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("p(chosen character)")
    fig.suptitle("Per-step certainty: dips are where the loop is guessing")
    fig.tight_layout()
    fig.savefig("results/chart_certainty.png", dpi=130)

    # 3. what the extra requests bought
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [q[1] for q in QUESTIONS]
    width = 0.38
    for offset, arm in zip((-width / 2, width / 2), ARMS):
        rows = [next(r for r in records if r["arm"] == arm and r["question"] == q[0])
                for q in QUESTIONS]
        bars = ax.bar([i + offset for i in range(len(labels))],
                      [int(r["requests"]) for r in rows], width,
                      color=colours[arm], label=arm)
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    "hit" if row["correct"] else "miss", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("requests")
    ax.set_title("Cost per answer, and whether it landed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/chart_cost.png", dpi=130)
    print("charts -> results/chart_logprob.png results/chart_certainty.png "
          "results/chart_cost.png")


if __name__ == "__main__":
    main()
