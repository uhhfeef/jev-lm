"""Beam search vs. the single-path loop, on the tie it was actually built for.

`probe_ensemble.py` showed the step after "j" is a genuine three-way tie --
`u=0.28 a=0.26 o=0.17`, converging tighter as orderings are averaged, so it is the
model's real belief and not position bias. `u` gives *jupiter* and the rest of the
word is near-forced (`p=0.66, i=0.69, r=0.96`); `a` and `o` give `janos` and
`jovis`, which stay flat. Lookahead should resolve exactly this.

Three runs per arm, different shuffles, scoring how often *jupiter* actually comes
out. The previous beam A/B ran one sample per cell and I read luck as a result;
reliability is the measurement here, not a single output.

Two differences from that earlier test, both deliberate:
  * the `jev` identity contamination is fixed, so the distribution is clean now
  * beams run at `length_penalty 0.7`, because at 1.0 they are biased toward
    stopping -- that is what produced `capital .` last time
The second means the arms differ in two ways, not one. That is the right trade
here: holding it at 1.0 would spend the budget re-measuring a configuration we
already know is broken.
"""

import json
import random
import time

from jev_lm import build_client, generate

RESULTS = "results/beam2_results.json"
QUESTION = "largest planet"
EXPECTED = "jupiter"
MAX_CHARS = 14
ENSEMBLE = 3
SEEDS = (1, 2, 3)          # matched across arms, so both see the same shuffles

ARMS = {
    "greedy": {"beam_width": 1, "length_penalty": 1.0},
    "beam3": {"beam_width": 3, "length_penalty": 0.7},
}


def main() -> None:
    records: list[dict[str, object]] = []
    started = time.monotonic()
    spent = 0

    with build_client() as client:
        for arm, cfg in ARMS.items():
            for seed in SEEDS:
                run = generate(
                    client, QUESTION, random.Random(seed),
                    max_chars=MAX_CHARS, quiet=True, window=40,
                    ensemble=ENSEMBLE,
                    beam_width=int(cfg["beam_width"]),
                    length_penalty=float(cfg["length_penalty"]),
                )
                spent += run.requests
                hit = EXPECTED in run.text
                records.append({
                    "arm": arm, "seed": seed, "text": run.text, "correct": hit,
                    "reason": run.reason, "requests": run.requests,
                    "steps": [{"c": s.char, "p": s.chosen_prob} for s in run.steps],
                })
                print(f"{'OK ' if hit else '   '}{arm:<8} seed={seed}  "
                      f"{run.requests:>3} req  {run.text!r}   [{spent} total]",
                      flush=True)
                with open(RESULTS, "w", encoding="utf-8") as handle:
                    json.dump(records, handle, indent=1)

    print(f"\n{spent} requests in {time.monotonic() - started:.0f}s -> {RESULTS}\n")
    print(f"{'arm':<8} {'hits':>7} {'requests':>9} {'outputs'}")
    print("-" * 64)
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        outs = ", ".join(sorted({str(r["text"]) for r in rows}))
        print(f"{arm:<8} {sum(bool(r['correct']) for r in rows):>4}/{len(rows)} "
              f"{sum(int(r['requests']) for r in rows):>9}  {outs}")


if __name__ == "__main__":
    main()
