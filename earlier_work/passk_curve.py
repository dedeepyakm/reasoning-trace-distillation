#!/usr/bin/env python3
"""
pass@k as a function of k, computed from existing trace files.

Answers the question raised in the meeting: why does the teacher-student gap
appear to shrink at pass@k?

Uses the unbiased estimator from Chen et al. (2021, Codex):

    pass@k = 1 - C(n-c, k) / C(n, k)

where n = samples generated per problem and c = number correct. This is the
standard estimator; naively taking "any correct in the first k" is biased and
high-variance.

Usage:
    python passk_curve.py traces_*.jsonl
"""

import json, sys, glob
from collections import defaultdict
from math import comb


def pass_at_k(n, c, k):
    """Unbiased pass@k for one problem: n samples, c correct."""
    if k > n:
        return None            # cannot estimate beyond samples available
    if n - c < k:
        return 1.0             # too few wrong ones to draw k all-wrong
    return 1.0 - comb(n - c, k) / comb(n, k)


def analyse(path):
    rows = [json.loads(l) for l in open(path)]
    model = rows[0].get("model", path)

    # group samples by problem
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["id"]].append(r)

    n_per = min(len(v) for v in by_problem.values())
    n_problems = len(by_problem)

    print("=" * 70)
    print(f"MODEL: {model}")
    print(f"  {n_problems} problems, {n_per} samples each")
    print("=" * 70)

    print(f"\n  {'k':>3}  {'pass@k':>8}   gain over pass@1")
    base = None
    for k in range(1, n_per + 1):
        vals = []
        for _, samples in by_problem.items():
            n = len(samples)
            c = sum(1 for s in samples if s["correct"])
            v = pass_at_k(n, c, k)
            if v is not None:
                vals.append(v)
        score = sum(vals) / len(vals)
        if base is None:
            base = score
        print(f"  {k:>3}  {score:>7.1%}   {score - base:>+8.1%}")

    # How many problems are "solvable but unreliable"?
    solved_always = sum(1 for v in by_problem.values()
                        if all(s["correct"] for s in v))
    solved_never = sum(1 for v in by_problem.values()
                       if not any(s["correct"] for s in v))
    solved_some = n_problems - solved_always - solved_never

    print(f"\n  problems always solved   : {solved_always:>3} ({solved_always/n_problems:.0%})")
    print(f"  problems never solved    : {solved_never:>3} ({solved_never/n_problems:.0%})")
    print(f"  problems sometimes solved: {solved_some:>3} ({solved_some/n_problems:.0%})"
          f"   <- pass@k grows from these")

    return {"model": model, "n_per": n_per,
            "always": solved_always / n_problems,
            "never": solved_never / n_problems,
            "some": solved_some / n_problems}


def main():
    paths = []
    for a in sys.argv[1:]:
        paths.extend(glob.glob(a))
    if not paths:
        print("usage: python passk_curve.py traces_*.jsonl")
        return

    summaries = [analyse(p) for p in sorted(paths)]

    print("\n" + "=" * 70)
    print("WHY THE GAP APPEARS TO SHRINK AT pass@k")
    print("=" * 70)
    print("""
  pass@k rises only from problems the model solves SOMETIMES. A model with
  many 'sometimes' problems gains a lot as k grows; one that is either
  reliably right or reliably wrong gains little.

  Note also that pass@k is NOT comparable across different k. Comparing a
  model measured at k=4 against one at k=2 gives the k=4 model more chances
  by construction. Any cross-model comparison must fix k.
""")
    print(f"  {'model':<46}{'sometimes':>11}{'samples':>9}")
    for s in summaries:
        name = s["model"].split("/")[-1]
        print(f"  {name:<46}{s['some']:>10.0%}{s['n_per']:>9}")


if __name__ == "__main__":
    main()
